# interrogator_chat.py

import asyncio
import logging
import re
import uuid

from datetime import datetime  # Added for conversation start time

from zmongo.zmongo_repository import ZMongoRepository as MongoRepository


# Configure logging
logging.basicConfig(level=logging.INFO)
CONVERSATIONS_COLLECTION = 'conversations'


class InterrogatorChat:
    def __init__(self, question_answer_pairs, _id, username):
        """
        Initialize the InterrogatorChat class.

        Args:
            question_answer_pairs (list): A list of dictionaries containing 'question' and 'expected_answer' keys.
            _id (str): Unique identifier for the conversation (e.g., user ID).
            username (str): The username of the authenticated user.
        """
        self.mongo_repo = MongoRepository()
        self.question_answer_pairs = question_answer_pairs
        self._id = _id
        self.username = username
        self.responses = []
        self.current_index = 0  # To keep track of the current question index

    def normalize_text(self, text):
        """
        Normalize text for comparison.

        Args:
            text (str): The text to normalize.

        Returns:
            str: Normalized text.
        """
        return re.sub(r'\s+', ' ', text.strip().lower())

    def confirm_response(self, user_response, expected_answer):
        """
        Confirm if the user's response matches the expected answer.

        Args:
            user_response (str): The user's response.
            expected_answer (str): The expected answer.

        Returns:
            bool: True if the response matches the expected answer, False otherwise.
        """
        # Normalize texts
        user_response_norm = self.normalize_text(user_response)
        expected_answer_norm = self.normalize_text(expected_answer)

        # Simple substring check; can be enhanced with NLP techniques
        if not expected_answer_norm:
            # If expected_answer is empty, accept any response
            return True
        return expected_answer_norm in user_response_norm

    async def save_response(self, question, response, expected_answer, confirmed, index):
        """
        Save the response to MongoDB.

        Args:
            question (str): The question asked.
            response (str): The user's response.
            expected_answer (str): The expected answer.
            confirmed (bool): Whether the response matches the expected answer.
            index (int): The index of the question in the list.
        """
        document = {
            '_id': self._id,
            'question_index': index,
            'question': question,
            'response': response,
            'expected_answer': expected_answer,
            'confirmed': confirmed,
            'creator': self.username  # Include creator field
        }
        try:
            # Upsert the response (update if exists, insert if not)
            await self.mongo_repo.update_document(
                collection='intake_responses',
                query={
                    '_id': self._id,
                    'question_index': index
                },
                update_data={'$set': document}
            )
            logging.info(f"Response saved to MongoDB: {document}")
        except Exception as e:
            logging.error(f"Error saving response to MongoDB: {e}")

    async def load_existing_responses(self):
        """
        Load existing responses from MongoDB.

        Returns:
            list: A list of existing responses sorted by question_index.
        """
        try:
            existing_responses = await self.mongo_repo.find_documents(
                collection='intake_responses',
                query={'_id': self._id},
                sort=[('question_index', 1)],
                limit=50  # Adjust limit as needed
            )
            return existing_responses
        except Exception as e:
            logging.error(f"Error loading existing responses: {e}")
            return []

    async def create_conversation_record(self):
        """
        Create a new conversation record with the username as the creator.
        """
        document = {
            '_id': self._id,
            'creator': self.username,
            'start_time': datetime.utcnow(),
        }
        try:
            await self.mongo_repo.insert_document(
                collection='conversations',  # You can use a separate collection for conversations
                document=document
            )
            logging.info(f"New conversation record created: {document}")
        except Exception as e:
            logging.error(f"Error creating conversation record: {e}")

    async def start_interrogation(self):
        """
        Start the interrogation process, allowing resumption of previous conversations.
        """
        print("Starting the intake process for eviction handling.\n")

        # Load existing responses
        existing_responses = await self.load_existing_responses()
        existing_indices = {resp['question_index']: resp for resp in existing_responses}

        # Ask if the user wants to resume
        if existing_responses:
            print("We found an existing conversation. Do you want to:")
            print("1. Resume where you left off")
            print("2. Review and update previous answers")
            print("3. Start over")
            choice = input("Enter your choice (1/2/3): ").strip()

            if choice == '2':
                await self.review_and_update(existing_responses)
            elif choice == '3':
                # Clear previous responses
                await self.mongo_repo.delete_document(
                    collection='intake_responses',
                    query={'_id': self._id},
                )
                existing_indices = {}
                print("Previous responses deleted. Starting over.\n")
                # Create a new conversation record since we're starting over
                await self.create_conversation_record()
            else:
                # Resume from where left off
                self.current_index = len(existing_responses)
                print(f"Resuming from question {self.current_index + 1}.\n")
        else:
            print("No previous conversation found. Starting fresh.\n")
            await self.create_conversation_record()  # Create a new conversation record

        # Continue with the remaining questions
        for index in range(self.current_index, len(self.question_answer_pairs)):
            qa = self.question_answer_pairs[index]
            question = qa['question']
            expected_answer = qa.get('expected_answer', '')

            # Ask the user the question
            print(f"Question: {question}")
            user_response = input("Your response: ")

            # Confirm if the user's response matches the expected answer
            confirmed = self.confirm_response(user_response, expected_answer)

            if confirmed:
                print("Response recorded.\n")
            else:
                print("Thank you for your response. We'll review it further.\n")

            # Save the response in MongoDB
            await self.save_response(
                question=question,
                response=user_response,
                expected_answer=expected_answer,
                confirmed=confirmed,
                index=index
            )

        # Close the MongoDB connection
        # await self.mongo_repo.close_connection()
        print("Intake process completed. Thank you!")

    async def review_and_update(self, existing_responses):
        """
        Allow the user to review and update previous responses.

        Args:
            existing_responses (list): List of existing responses.
        """
        print("\nReviewing your previous responses:\n")
        for resp in existing_responses:
            index = resp['question_index']
            question = resp['question']
            response = resp['response']
            expected_answer = resp.get('expected_answer', '')
            confirmed = resp['confirmed']

            print(f"Question: {question}")
            print(f"Your previous response: {response}")
            update_choice = input("Do you want to update this response? (y/n): ").strip().lower()

            if update_choice == 'y':
                user_response = input("Enter your new response: ")
                # Confirm if the user's response matches the expected answer
                confirmed = self.confirm_response(user_response, expected_answer)

                if confirmed:
                    print("Response recorded.\n")
                else:
                    print("Thank you for your response. We'll review it further.\n")

                # Save the updated response
                await self.save_response(
                    question=question,
                    response=user_response,
                    expected_answer=expected_answer,
                    confirmed=confirmed,
                    index=index
                )
            else:
                print("Keeping the previous response.\n")

        # Set the current index to the last reviewed question
        self.current_index = existing_responses[-1]['question_index'] + 1
        print("Finished reviewing previous responses.\n")


# Sample question/expected answer pairs
these_question_answer_pairs = [
    {
        'question': 'What is your full name?',
        'expected_answer': ''  # Leave blank if any answer is acceptable
    },
    {
        'question': 'What is the address of the property in question?',
        'expected_answer': ''
    },
    {
        'question': 'Is the property a commercial or residential property?',
        'expected_answer': 'commercial'  # or 'residential'
    },
    {
        'question': 'What is the name of the tenant you wish to evict?',
        'expected_answer': ''
    },
    {
        'question': 'What is the reason for eviction?',
        'expected_answer': ''
    },
    {
        'question': 'How long has the tenant been in default?',
        'expected_answer': ''
    },
    {
        'question': 'Have you served any notices to the tenant? If so, what type?',
        'expected_answer': ''
    },
    {
        'question': 'Do you have a written lease agreement with the tenant?',
        'expected_answer': 'yes'  # or 'no'
    },
    {
        'question': 'Are there any other issues or damages caused by the tenant?',
        'expected_answer': ''
    },
    {
        'question': 'What outcome are you seeking from this eviction process?',
        'expected_answer': ''
    }
]

import asyncio
import uuid
import logging

from zmongo.zmongo_repository import ZMongoRepository as MongoRepository
from datetime import datetime

CONVERSATIONS_COLLECTION = "conversations"
logging.basicConfig(level=logging.INFO)

async def main():
    print("Welcome to the Eviction Intake Form.")

    mongo_repo = MongoRepository()

    # Authentication and conversation initialization
    while True:
        username = input("Please enter your username: ").strip()
        email = input("Please enter your email: ").strip()

        # Verify user exists in the 'user' collection
        user = await mongo_repo.find_document(
            collection="user",
            query={"username": username, "email": email}
        )

        if user:
            print("Authentication successful.\n")

            # Check if user has existing conversations
            users_conversations = await mongo_repo.find_documents(
                collection=CONVERSATIONS_COLLECTION,
                query={"creator": username},
                sort=[("start_time", -1)]  # Sort by most recent conversation
            )

            if users_conversations:
                print(f"Welcome back, {username}! You have {len(users_conversations)} existing conversation(s).\n")
                print("Do you want to:")
                print("1. Resume your most recent conversation")
                print("2. Start a new conversation")
                choice = input("Enter your choice (1/2): ").strip()

                if choice == "1":
                    # Use the most recent conversation
                    this_conversation = users_conversations[0]
                    _id = str(this_conversation["_id"])
                    print(f"Resuming conversation ID: {_id}\n")
                else:
                    # Create a new conversation
                    _id = await create_new_conversation(mongo_repo, username)
                    print(f"New conversation started with ID: {_id}\n")
            else:
                # No existing conversations, create a new one
                print(f"Welcome, {username}! Starting your first conversation.\n")
                _id = await create_new_conversation(mongo_repo, username)

            # Exit the authentication loop
            break
        else:
            print("Invalid username or email. Please try again.\n")

    # Define the question-answer pairs
    question_answer_pairs = [
        {"question": "What is your full name?", "expected_answer": ""},
        {"question": "What is the address of the property in question?", "expected_answer": ""},
        {"question": "Is the property a commercial or residential property?", "expected_answer": ""},
        {"question": "What is the name of the tenant you wish to evict?", "expected_answer": ""},
        {"question": "What is the reason for eviction?", "expected_answer": ""},
        {"question": "How long has the tenant been in default?", "expected_answer": ""},
        {"question": "Have you served any notices to the tenant? If so, what type?", "expected_answer": ""},
        {"question": "Do you have a written lease agreement with the tenant?", "expected_answer": ""},
        {"question": "Are there any other issues or damages caused by the tenant?", "expected_answer": ""},
        {"question": "What outcome are you seeking from this eviction process?", "expected_answer": ""}
    ]

    # Start the interrogation process
    chat = InterrogatorChat(question_answer_pairs, _id, username)
    await chat.start_interrogation()

async def create_new_conversation(mongo_repo, username):
    """
    Create a new conversation record in the database.

    Args:
        mongo_repo (MongoRepository): The MongoDB repository instance.
        username (str): The username of the authenticated user.

    Returns:
        str: The ID of the newly created conversation.
    """
    new__id = uuid.uuid4().hex
    conversation_document = {
        "_id": new__id,  # Use the generated UUID as the MongoDB _id
        "creator": username,
        "start_time": datetime.utcnow()
    }

    await mongo_repo.insert_document(
        collection=CONVERSATIONS_COLLECTION,
        document=conversation_document
    )

    logging.info(f"New conversation created: {conversation_document}")
    return new__id

if __name__ == "__main__":
    asyncio.run(main())
