import os
import uuid
from typing import Optional, List
from dotenv import load_dotenv
from rich import print
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import create_extraction_chain_pydantic
from langchain import hub

load_dotenv()


class StatementCollection(BaseModel):
    """
    Pydantic schema for extracting structured statements/propositions from text.
    Used for proposition-based text processing.
    """
    statements: List[str] = Field(description="List of extracted statements or propositions")


class PropositionExtractor:
    """
    Utility class for extracting propositions/statements from raw text.
    Uses LangChain Hub prompts and LLM-based extraction to break down text
    into individual propositions suitable for semantic chunking.
    """
    
    def __init__(self, openai_api_key=None, hub_prompt_identifier="wfh/proposal-indexing"):
        """
        Initialize the proposition extractor.
        
        Args:
            openai_api_key: OpenAI API key (optional, will use environment variable if not provided)
            hub_prompt_identifier: LangChain Hub prompt identifier for proposition extraction
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or environment variable")
        
        self.llm_model = ChatOpenAI(
            model='gpt-3.5-turbo',
            openai_api_key=api_key,
            temperature=0
        )
        
        # Load prompt from LangChain Hub
        try:
            self.hub_prompt = hub.pull(hub_prompt_identifier)
            self.proposition_chain = self.hub_prompt | self.llm_model
        except Exception as e:
            raise ValueError(f"Failed to load prompt from LangChain Hub ({hub_prompt_identifier}): {str(e)}")
        
        # Create extraction chain for structured output
        self.extraction_chain = create_extraction_chain_pydantic(
            pydantic_schema=StatementCollection,
            llm=self.llm_model
        )
    
    def extract_statements(self, text):
        """
        Extract individual statements/propositions from a given text.
        
        Args:
            text: Raw text string to extract propositions from
        
        Returns:
            List of extracted statements/propositions
        """
        # Use hub prompt to process text
        llm_output = self.proposition_chain.invoke({"input": text}).content
        
        # Extract structured statements using Pydantic schema
        extraction_result = self.extraction_chain.invoke(llm_output)["text"]
        
        if extraction_result and len(extraction_result) > 0:
            return extraction_result[0].statements
        else:
            return []
    
    def extract_from_paragraphs(self, paragraphs, max_paragraphs=None, verbose=False):
        """
        Extract statements from multiple paragraphs.
        
        Args:
            paragraphs: List of paragraph strings or a single text with paragraph separators
            max_paragraphs: Maximum number of paragraphs to process (None for all)
            verbose: Whether to print progress messages
        
        Returns:
            List of all extracted statements from all paragraphs
        """
        # Handle both list and string input
        if isinstance(paragraphs, str):
            paragraph_list = paragraphs.split("\n\n")
        else:
            paragraph_list = paragraphs
        
        # Limit paragraphs if specified
        if max_paragraphs:
            paragraph_list = paragraph_list[:max_paragraphs]
        
        all_statements = []
        
        for idx, paragraph in enumerate(paragraph_list):
            if not paragraph.strip():
                continue
                
            statements = self.extract_statements(paragraph)
            all_statements.extend(statements)
            
            if verbose:
                print(f"Processed paragraph {idx + 1}/{len(paragraph_list)}: Extracted {len(statements)} statements")
        
        return all_statements


class SemanticChunkManager:
    """
    Intelligent semantic chunking system that groups related statements
    into coherent chunks using LLM-based semantic understanding.
    """
    
    def __init__(self, openai_api_key=None):
        # Storage for semantic groups
        self.semantic_groups = {}
        
        # Configuration parameters
        self.group_id_length = 5
        self.auto_refine_metadata = True
        self.enable_verbose_output = True
        
        # Initialize LLM connection
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or environment variable")
        
        self.llm_model = ChatOpenAI(
            model='gpt-3.5-turbo',
            openai_api_key=api_key,
            temperature=0
        )

    def process_statements(self, statements):
        """Process multiple statements and group them semantically."""
        for statement in statements:
            self.process_statement(statement)
    
    def process_text(self, text, extractor=None, max_paragraphs=None):
        """
        Process raw text by extracting propositions and then grouping them semantically.
        
        Args:
            text: Raw text string or list of paragraphs to process
            extractor: PropositionExtractor instance (will create one if not provided)
            max_paragraphs: Maximum number of paragraphs to process (None for all)
        
        Returns:
            List of extracted statements before chunking
        """
        if extractor is None:
            # Create extractor using same API key as chunk manager
            api_key = os.getenv("OPENAI_API_KEY")
            extractor = PropositionExtractor(openai_api_key=api_key)
        
        # Extract statements from text
        if self.enable_verbose_output:
            print("Extracting statements from text...")
        
        statements = extractor.extract_from_paragraphs(
            text, 
            max_paragraphs=max_paragraphs,
            verbose=self.enable_verbose_output
        )
        
        if self.enable_verbose_output:
            print(f"Extracted {len(statements)} statements. Processing for semantic grouping...")
        
        # Process extracted statements
        self.process_statements(statements)
        
        return statements

    def process_statement(self, statement):
        """Process a single statement and assign it to an appropriate semantic group."""
        if self.enable_verbose_output:
            print(f"\nProcessing statement: '{statement}'")

        # Handle the first statement - create initial group
        if not self.semantic_groups:
            if self.enable_verbose_output:
                print("Initializing first semantic group")
            self._initialize_new_group(statement)
            return

        # Search for semantically related existing group
        matching_group_id = self._locate_semantic_match(statement)

        if matching_group_id:
            if self.enable_verbose_output:
                group_info = self.semantic_groups[matching_group_id]
                print(f"Match found (Group {group_info['group_id']}): {group_info['group_title']}")
            self._append_to_group(matching_group_id, statement)
        else:
            if self.enable_verbose_output:
                print("No matching group found, creating new group")
            self._initialize_new_group(statement)

    def _append_to_group(self, group_id, statement):
        """Add a statement to an existing group and update metadata if enabled."""
        self.semantic_groups[group_id]['statements'].append(statement)

        if self.auto_refine_metadata:
            group_data = self.semantic_groups[group_id]
            group_data['group_summary'] = self._refine_group_summary(group_data)
            group_data['group_title'] = self._refine_group_title(group_data)

    def _refine_group_summary(self, group_data):
        """
        Update the summary of a group when new statements are added.
        Ensures summaries remain accurate as groups evolve.
        """
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You manage semantic groups that contain related statements sharing common themes or topics.
                When a new statement is added to a group, generate a concise one-sentence summary that 
                accurately describes what the group represents.

                The summary should clearly indicate the group's topic and provide guidance on what 
                types of statements belong in this group.

                You will receive all statements currently in the group along with the existing summary.

                Apply generalization principles: specific items should be generalized to broader categories.
                For instance, "apples" becomes "food items", and "October" becomes "temporal information".

                Example:
                Statement: Greg enjoys eating pizza
                Summary: This group contains information about Greg's food preferences and dietary choices.

                Provide only the updated summary, no additional text.
                """
            ),
            (
                "user",
                "Group statements:\n{statements}\n\nExisting summary:\n{existing_summary}"
            )
        ])

        prompt_chain = summary_prompt | self.llm_model

        updated_summary = prompt_chain.invoke({
            "statements": "\n".join(group_data['statements']),
            "existing_summary": group_data['group_summary']
        }).content

        return updated_summary

    def _refine_group_title(self, group_data):
        """
        Update the title of a group when new statements are added.
        Keeps titles current and representative of group content.
        """
        title_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You manage semantic groups that contain related statements sharing common themes or topics.
                When a new statement is added to a group, generate a brief, descriptive title that 
                captures the essence of what the group represents.

                The title should be concise yet comprehensive enough to indicate the group's focus.

                You will receive all statements in the group, the current summary, and the existing title.

                Apply generalization principles: specific items should be generalized to broader categories.
                For instance, "apples" becomes "food items", and "October" becomes "temporal information".

                Example:
                Summary: This group contains temporal information and date-related statements
                Title: Temporal Information

                Provide only the updated title, no additional text.
                """
            ),
            (
                "user",
                "Group statements:\n{statements}\n\nGroup summary:\n{group_summary}\n\nCurrent title:\n{current_title}"
            )
        ])

        prompt_chain = title_prompt | self.llm_model

        updated_title = prompt_chain.invoke({
            "statements": "\n".join(group_data['statements']),
            "group_summary": group_data['group_summary'],
            "current_title": group_data['group_title']
        }).content

        return updated_title

    def _generate_initial_summary(self, statement):
        """Generate the initial summary for a newly created group."""
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You manage semantic groups that contain related statements sharing common themes or topics.
                Generate a concise one-sentence summary that describes what a new group will represent.

                The summary should clearly indicate the group's topic and provide guidance on what 
                types of statements belong in this group.

                You will receive a single statement that will be the first item in a new group.

                Apply generalization principles: specific items should be generalized to broader categories.
                For instance, "apples" becomes "food items", and "October" becomes "temporal information".

                Example:
                Statement: Greg enjoys eating pizza
                Summary: This group contains information about Greg's food preferences and dietary choices.

                Provide only the new summary, no additional text.
                """
            ),
            (
                "user",
                "Generate a summary for a new group containing this statement:\n{statement}"
            )
        ])

        prompt_chain = summary_prompt | self.llm_model

        initial_summary = prompt_chain.invoke({
            "statement": statement
        }).content

        return initial_summary

    def _generate_initial_title(self, summary):
        """Generate the initial title for a newly created group based on its summary."""
        title_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You manage semantic groups that contain related statements sharing common themes or topics.
                Generate a brief, descriptive title that captures the essence of what a group represents.

                The title should be concise yet comprehensive enough to indicate the group's focus.

                You will receive the summary of a group that needs a title.

                Apply generalization principles: specific items should be generalized to broader categories.
                For instance, "apples" becomes "food items", and "October" becomes "temporal information".

                Example:
                Summary: This group contains temporal information and date-related statements
                Title: Temporal Information

                Provide only the new title, no additional text.
                """
            ),
            (
                "user",
                "Generate a title for a group with this summary:\n{summary}"
            )
        ])

        prompt_chain = title_prompt | self.llm_model

        initial_title = prompt_chain.invoke({
            "summary": summary
        }).content

        return initial_title

    def _initialize_new_group(self, statement):
        """Create a new semantic group with the given statement."""
        new_group_id = str(uuid.uuid4())[:self.group_id_length]
        initial_summary = self._generate_initial_summary(statement)
        initial_title = self._generate_initial_title(initial_summary)

        self.semantic_groups[new_group_id] = {
            'group_id': new_group_id,
            'statements': [statement],
            'group_title': initial_title,
            'group_summary': initial_summary,
            'group_index': len(self.semantic_groups)
        }

        if self.enable_verbose_output:
            print(f"New group created ({new_group_id}): {initial_title}")

    def retrieve_group_overview(self):
        """
        Generate a formatted overview of all existing semantic groups.
        Returns an empty string if no groups exist yet.
        """
        overview_text = ""

        for group_id, group_info in self.semantic_groups.items():
            group_entry = f"""Group ({group_info['group_id']}): {group_info['group_title']}\nSummary: {group_info['group_summary']}\n\n"""
            overview_text += group_entry

        return overview_text

    def _locate_semantic_match(self, statement):
        """
        Use LLM to determine if a statement belongs to any existing semantic group.
        Returns the group ID if a match is found, None otherwise.
        """
        groups_overview = self.retrieve_group_overview()

        matching_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Evaluate whether the provided statement semantically belongs to any of the existing groups.

                A statement belongs to a group if their semantic meaning, thematic direction, or 
                conceptual intention align. The objective is to cluster related statements together.

                If the statement matches an existing group, return the group's ID.
                If no suitable match exists, return "No matching group".

                Example:
                Statement: "Greg really likes hamburgers"
                Existing Groups:
                    - Group ID: 2n4l3d
                    - Group Title: Places in San Francisco
                    - Group Summary: Overview of the things to do with San Francisco Places

                    - Group ID: 93833k
                    - Group Title: Food Greg likes
                    - Group Summary: Lists of the food and dishes that Greg likes
                Response: 93833k
                """
            ),
            (
                "user",
                "Existing Groups:\n---Begin groups---\n{groups_overview}---End groups---"
            ),
            (
                "user",
                "Evaluate if this statement belongs to any of the groups above:\n{statement}"
            )
        ])

        prompt_chain = matching_prompt | self.llm_model

        llm_response = prompt_chain.invoke({
            "statement": statement,
            "groups_overview": groups_overview
        }).content

        # Extract group ID using structured extraction
        class GroupIdentifier(BaseModel):
            """Schema for extracting group identifier"""
            group_id: Optional[str]

        extraction_chain = create_extraction_chain_pydantic(
            pydantic_schema=GroupIdentifier,
            llm=self.llm_model
        )
        
        extracted_result = extraction_chain.invoke(llm_response)["text"]
        if extracted_result:
            llm_response = extracted_result[0].group_id

        # Validate response length matches expected group ID format
        if len(llm_response) != self.group_id_length:
            return None

        return llm_response

    def retrieve_groups(self, output_format='dict'):
        """
        Retrieve semantic groups in the specified format.
        
        Args:
            output_format: 'dict' returns full dictionary structure,
                          'list_of_strings' returns list of concatenated statement strings
        
        Returns:
            Groups in the requested format
        """
        if output_format == 'dict':
            return self.semantic_groups
        
        if output_format == 'list_of_strings':
            group_strings = []
            for group_id, group_info in self.semantic_groups.items():
                combined_statements = " ".join(group_info['statements'])
                group_strings.append(combined_statements)
            return group_strings

    def display_groups(self):
        """Display all semantic groups in a formatted, readable manner."""
        print(f"\nTotal groups: {len(self.semantic_groups)}\n")
        
        for group_id, group_info in self.semantic_groups.items():
            print(f"Group #{group_info['group_index']}")
            print(f"Group ID: {group_id}")
            print(f"Summary: {group_info['group_summary']}")
            print("Statements:")
            for stmt in group_info['statements']:
                print(f"    - {stmt}")
            print("\n\n")

    def display_group_overview(self):
        """Display a formatted overview of all semantic groups."""
        print("Semantic Groups Overview\n")
        print(self.retrieve_group_overview())
    
    def to_documents(self, metadata=None):
        """
        Convert semantic groups to LangChain Document objects for RAG integration.
        
        Args:
            metadata: Optional base metadata dictionary to add to all documents
        
        Returns:
            List of Document objects, one per semantic group
        """
        try:
            from langchain.docstore.document import Document
        except ImportError:
            raise ImportError(
                "langchain.docstore.document is required for document conversion. "
                "Install with: pip install langchain"
            )
        
        documents = []
        base_metadata = metadata or {}
        
        for group_id, group_info in self.semantic_groups.items():
            # Combine all statements in the group
            content = " ".join(group_info['statements'])
            
            # Create document with group metadata
            doc_metadata = {
                **base_metadata,
                'group_id': group_id,
                'group_title': group_info['group_title'],
                'group_summary': group_info['group_summary'],
                'group_index': group_info['group_index'],
                'statement_count': len(group_info['statements'])
            }
            
            documents.append(Document(page_content=content, metadata=doc_metadata))
        
        return documents


if __name__ == "__main__":
    print("=" * 80)
    print("Semantic Chunk Manager - Comprehensive Example")
    print("=" * 80)
    
    # Example 1: Direct statement processing
    print("\n[Example 1] Processing statements directly\n")
    chunk_manager = SemanticChunkManager()

    test_statements = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
        # Additional statements can be uncommented for testing
        # 'Teachers and coaches meant well.',
        # "The statement that 'You get out what you put in' is rarely true.",
        # "If your product is only half as good as your competitor's product, you do not get half as many customers.",
        # "You get no customers if your product is only half as good as your competitor's product.",
        # 'You go out of business if you get no customers.',
        # 'The returns for performance are superlinear in business.',
        # 'Some people think the superlinear returns for performance are a flaw of capitalism.',
        # 'Some people think that changing the rules of capitalism would stop the superlinear returns for performance from being true.',
        # 'Superlinear returns for performance are a feature of the world.',
        # 'Superlinear returns for performance are not an artifact of rules that humans have invented.',
        # 'The same pattern of superlinear returns is observed in fame.',
        # 'The same pattern of superlinear returns is observed in power.',
        # 'The same pattern of superlinear returns is observed in military victories.',
        # 'The same pattern of superlinear returns is observed in knowledge.',
        # 'The same pattern of superlinear returns is observed in benefit to humanity.',
        # 'In fame, power, military victories, knowledge, and benefit to humanity, the rich get richer.'
    ]

    chunk_manager.process_statements(test_statements)
    chunk_manager.display_groups()
    chunk_manager.display_group_overview()
    
    print("\n" + "=" * 80)
    print("[Example 2] Using Proposition Extractor for raw text processing")
    print("=" * 80)
    
    # Example 2: Extract propositions from raw text
    sample_text = """
    Text splitting in LangChain is a critical feature that facilitates the division of large texts into smaller, manageable segments. 
    This process is essential for working with language models that have token limits.
    
    Recursive Character Text Splitting is a technique used in Natural Language Processing to break down large text documents 
    into smaller chunks by repeatedly splitting the text based on specific characters, like newlines or spaces, while 
    prioritizing keeping semantically related content together.
    
    Semantic Chunking considers the relationships within the text. It divides the text into meaningful, semantically 
    complete chunks. This approach ensures the information's integrity during retrieval, leading to more accurate results.
    """
    
    print("\nExtracting propositions from sample text...")
    extractor = PropositionExtractor()
    extracted_statements = extractor.extract_from_paragraphs(
        sample_text, 
        max_paragraphs=3,
        verbose=True
    )
    
    print(f"\nExtracted {len(extracted_statements)} statements:")
    for i, stmt in enumerate(extracted_statements[:5], 1):  # Show first 5
        print(f"  {i}. {stmt}")
    if len(extracted_statements) > 5:
        print(f"  ... and {len(extracted_statements) - 5} more")
    
    # Example 3: Process text directly with chunk manager
    print("\n" + "=" * 80)
    print("[Example 3] Processing raw text directly with SemanticChunkManager")
    print("=" * 80)
    
    chunk_manager2 = SemanticChunkManager()
    chunk_manager2.process_text(sample_text, max_paragraphs=2)
    chunk_manager2.display_groups()
    
    # Example 4: Convert to Document objects for RAG
    print("\n" + "=" * 80)
    print("[Example 4] Converting groups to LangChain Documents for RAG")
    print("=" * 80)
    
    try:
        documents = chunk_manager2.to_documents(metadata={"source": "example_text"})
        print(f"\nCreated {len(documents)} Document objects:")
        for doc in documents:
            print(f"  - Group: {doc.metadata['group_title']}")
            print(f"    Content length: {len(doc.page_content)} characters")
            print(f"    Statements: {doc.metadata['statement_count']}")
    except ImportError as e:
        print(f"\nNote: Document conversion requires langchain: {e}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
