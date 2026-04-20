#!/usr/bin/env python3
"""
Stage 11b: Retrieve Human Evaluation Responses

Retrieves responses from Google Forms created in Stage 11.
Downloads all responses and matches them to questions for analysis.

Usage: python 11b_retrieve_responses.py
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

# Google Forms API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from utils.files import write_json_file
from utils.logging import get_stage_logger, log_stage_start, log_stage_end

# ============================================================================
# Configuration
# ============================================================================

# Google Forms API Configuration (with responses scope added)
SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
CLIENT_SECRETS = "/data3/zkachwal/reddit-mod-collection-pipeline/credentials/client_secret_795576073496-qo2r4ntgn1drrqo31p98it9bmtd2hvm4.apps.googleusercontent.com.json"
TOKEN_FILE = "/data3/zkachwal/reddit-mod-collection-pipeline/credentials/token.json"


# ============================================================================
# Helper Functions
# ============================================================================

def authenticate():
    """Authenticate with Google APIs using OAuth2."""
    creds = None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    except Exception:
        pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save for next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds


# ============================================================================
# Load Metadata
# ============================================================================

def load_evaluation_metadata(logger) -> Dict[str, Any]:
    """Load the evaluation metadata created in Stage 11."""
    metadata_file = os.path.join(PATHS['data'], 'evaluation', 'stage11_human_evaluation_metadata.json')

    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return None

    logger.info(f"Loading metadata from: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata for {len(metadata['questions'])} questions")
    return metadata


# ============================================================================
# Retrieve Responses
# ============================================================================

def get_form_structure(service, form_id: str, logger) -> Dict[str, Any]:
    """Get the form structure to map question IDs to indices."""
    try:
        form = service.forms().get(formId=form_id).execute()
        return form
    except Exception as e:
        logger.error(f"Error getting form structure for {form_id}: {e}")
        return None


def get_form_responses(service, form_id: str, logger) -> List[Dict]:
    """Retrieve all responses for a given form."""
    try:
        result = service.forms().responses().list(formId=form_id).execute()
        responses = result.get('responses', [])
        logger.info(f"Retrieved {len(responses)} responses from form {form_id}")
        return responses
    except Exception as e:
        logger.error(f"Error retrieving responses for {form_id}: {e}")
        return []


def parse_response(response: Dict, form_structure: Dict, question_offset: int) -> Dict[str, Any]:
    """Parse a single form response into structured format.

    Args:
        response: Response from Google Forms API
        form_structure: Form structure with question mapping
        question_offset: Offset to add to question indices (0 for form 1, 50 for form 2)

    Returns:
        Dict mapping question_index to list of selected answers
    """
    parsed = {
        'response_id': response.get('responseId'),
        'create_time': response.get('createTime'),
        'last_submitted_time': response.get('lastSubmittedTime'),
        'answers': {}
    }

    # Build question ID to index mapping from form structure
    question_id_to_index = {}
    items = form_structure.get('items', [])

    # Track question index (excluding page breaks)
    question_idx = 0
    for item in items:
        if 'questionItem' in item:
            # Use questionId from the questionItem, not itemId
            question_id = item['questionItem']['question'].get('questionId')
            # Add offset for part 2 forms
            question_id_to_index[question_id] = question_offset + question_idx + 1
            question_idx += 1

    # Parse answers
    answers = response.get('answers', {})
    for question_id, answer_data in answers.items():
        if question_id in question_id_to_index:
            question_index = question_id_to_index[question_id]

            # Extract text answers (for checkbox questions)
            text_answers = answer_data.get('textAnswers', {})
            answer_list = [ans.get('value') for ans in text_answers.get('answers', [])]

            # Use string key for JSON serialization
            parsed['answers'][str(question_index)] = answer_list

    return parsed


def retrieve_all_responses(service, metadata: Dict, logger) -> Dict[str, List[Dict]]:
    """Retrieve and parse all responses from both forms.

    Returns:
        Dict with 'form_1' and 'form_2' keys, each containing list of responses
    """
    responses_by_form = {'form_1': [], 'form_2': []}

    for form_info in metadata['forms']:
        form_id = form_info['form_id']
        form_part = form_info['form_part']
        question_offset = (form_part - 1) * 50  # 0 for part 1, 50 for part 2

        logger.info(f"\nRetrieving responses for Form Part {form_part}...")

        # Get form structure
        form_structure = get_form_structure(service, form_id, logger)
        if not form_structure:
            continue

        # Get responses
        responses = get_form_responses(service, form_id, logger)

        # Parse each response
        form_key = f'form_{form_part}'
        for response in responses:
            parsed = parse_response(response, form_structure, question_offset)
            responses_by_form[form_key].append(parsed)

    return responses_by_form


# ============================================================================
# Process and Save
# ============================================================================

def aggregate_responses(responses_by_form: Dict[str, List[Dict]], metadata: Dict) -> Dict[str, Any]:
    """Aggregate responses by annotator and question.

    Merges responses from the same annotator across both forms.
    Assumes form_1 and form_2 responses are in the same order (annotator 1 is first in both, etc.)
    """
    from collections import Counter

    # Merge responses from same annotators across forms
    # Assumption: responses[0] from form_1 and responses[0] from form_2 are the same person
    form_1_responses = responses_by_form.get('form_1', [])
    form_2_responses = responses_by_form.get('form_2', [])

    num_annotators = max(len(form_1_responses), len(form_2_responses))

    annotators = {}
    for idx in range(num_annotators):
        annotator_id = f"annotator_{idx + 1}"
        annotators[annotator_id] = {
            'response_ids': [],
            'answers': {}
        }

        # Merge form 1 response if exists
        if idx < len(form_1_responses):
            resp = form_1_responses[idx]
            annotators[annotator_id]['response_ids'].append(resp['response_id'])
            annotators[annotator_id]['answers'].update(resp['answers'])

        # Merge form 2 response if exists
        if idx < len(form_2_responses):
            resp = form_2_responses[idx]
            annotators[annotator_id]['response_ids'].append(resp['response_id'])
            annotators[annotator_id]['answers'].update(resp['answers'])

    # Build question-level data
    questions_with_responses = []
    for question in metadata['questions']:
        question_idx = question['question_index']

        question_data = {
            **question,  # Include all original metadata
            'human_annotations': []
        }

        # Collect annotations from all annotators
        for annotator_id, annotator_data in annotators.items():
            # Convert question_idx to string for lookup
            if str(question_idx) in annotator_data['answers']:
                question_data['human_annotations'].append({
                    'annotator_id': annotator_id,
                    'selected_answers': annotator_data['answers'][str(question_idx)]
                })

        # Compute majority vote: count each answer selection as a vote
        # A rule gets majority if ≥2 annotators selected it
        answer_votes = Counter()
        total_annotators = len(question_data['human_annotations'])

        for annotation in question_data['human_annotations']:
            for answer in annotation['selected_answers']:
                answer_votes[answer] += 1

        # Find answers with majority (≥2 votes)
        majority_answers = [answer for answer, count in answer_votes.items() if count >= 2]

        # Store majority voting results
        question_data['majority_answers'] = majority_answers if majority_answers else []
        question_data['answer_vote_counts'] = dict(answer_votes)
        question_data['num_annotators'] = total_annotators

        questions_with_responses.append(question_data)

    result = {
        'metadata': metadata['metadata'],
        'forms': metadata['forms'],
        'distributions': metadata['distributions'],
        'num_annotators': len(annotators),
        'annotators': annotators,
        'questions': questions_with_responses
    }

    return result


def save_responses(data: Dict, output_dir: str, logger) -> str:
    """Save the retrieved responses."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'stage11_human_annotations.json')
    write_json_file(data, output_file, pretty=True)
    logger.info(f"Saved human annotations to: {output_file}")

    return output_file


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(11, "retrieve_responses")
    log_stage_start(logger, 11, "Retrieve Human Evaluation Responses")

    start_time = time.time()

    try:
        print("=" * 60)
        print("Stage 11b: Retrieve Human Evaluation Responses")
        print("=" * 60)

        # Load metadata
        print(f"\n{'='*60}")
        print("LOADING METADATA")
        print("=" * 60)
        metadata = load_evaluation_metadata(logger)
        if not metadata:
            logger.error("Failed to load metadata!")
            return 1

        # Authenticate
        print(f"\n{'='*60}")
        print("AUTHENTICATING")
        print("=" * 60)
        print("Authenticating with Google APIs...")
        creds = authenticate()
        service = build('forms', 'v1', credentials=creds)
        print("Authenticated successfully")

        # Retrieve responses
        print(f"\n{'='*60}")
        print("RETRIEVING RESPONSES")
        print("=" * 60)
        responses_by_form = retrieve_all_responses(service, metadata, logger)

        total_responses = len(responses_by_form.get('form_1', [])) + len(responses_by_form.get('form_2', []))
        if total_responses == 0:
            logger.warning("No responses found!")
            return 1

        print(f"\nTotal responses retrieved: {total_responses}")
        print(f"  Form Part 1: {len(responses_by_form.get('form_1', []))} responses")
        print(f"  Form Part 2: {len(responses_by_form.get('form_2', []))} responses")

        # Aggregate and process
        print(f"\n{'='*60}")
        print("PROCESSING RESPONSES")
        print("=" * 60)
        aggregated = aggregate_responses(responses_by_form, metadata)

        # Save
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print("=" * 60)
        output_dir = os.path.join(PATHS['data'], 'evaluation')
        output_file = save_responses(aggregated, output_dir, logger)

        # Compute majority voting statistics
        questions_with_majority = sum(1 for q in aggregated['questions'] if q['majority_answers'])
        questions_without_majority = len(aggregated['questions']) - questions_with_majority

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"Number of annotators: {aggregated['num_annotators']}")
        print(f"Number of questions: {len(aggregated['questions'])}")
        print(f"\nAnnotator details:")
        for annotator_id, data in aggregated['annotators'].items():
            num_answered = len(data['answers'])
            print(f"  {annotator_id}: {num_answered} questions answered")
        print(f"\nMajority voting results:")
        print(f"  Questions WITH majority (≥2 votes): {questions_with_majority}")
        print(f"  Questions WITHOUT majority: {questions_without_majority}")
        print(f"\nResults saved to: {output_file}")

        elapsed = time.time() - start_time
        print(f"\nStage 11b completed in {elapsed:.1f}s")
        log_stage_end(logger, 11, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"Stage 11b failed: {e}")
        import traceback
        traceback.print_exc()
        log_stage_end(logger, 11, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
