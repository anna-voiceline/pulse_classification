import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import argparse


def load_data(texts_path, embeddings_path):
    """
    Load the text data and embeddings from specified paths

    Args:
        texts_path: Path to the text data file (.npy)
        embeddings_path: Path to the embeddings file (.npy)

    Returns:
        texts: NumPy array of text data
        embeddings: NumPy array of embeddings
    """
    try:
        texts = np.load(texts_path, allow_pickle=True)
        embeddings = np.load(embeddings_path, allow_pickle=True)
        print(f"Loaded {len(texts)} text samples and {len(embeddings)} embeddings")
        return texts, embeddings
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def identify_call_visit_patterns(texts):
    """
    Use pattern matching to identify texts related to calls or visits

    Args:
        texts: NumPy array of text data

    Returns:
        dirty_indices: List of indices that likely contain call or visit reports
    """
    # German and English patterns for calls and visits
    # Will be applied case-insensitively
    call_patterns = [
        # German call-related patterns
        r'\btelefonat\b',  # telephone call
        r'\btelefon\b',  # telephone
        r'\banruf\b',  # call
        r'\bangerufen\b',  # called
        r'\btelefoniert\b',  # telephoned
        r'\branruf\b',  # callback
        r'\brückruf\b',  # return call
        r'\btelefonisch\b',  # by telephone
        r'\btelefonkonferenz\b',  # telephone conference
        r'\btelefonisch kontaktiert\b',  # contacted by phone

        # English call-related patterns
        r'\bcall\b',  # call
        r'\bphone\b',  # phone
        r'\btelephon',  # telephone
        r'\bcalled\b',  # called
        r'\bcalling\b',  # calling
        r'\bphoned\b',  # phoned
        r'\bdialed\b',  # dialed
        r'\breached out\b',  # reached out
        r'\bgot in touch\b',  # got in touch
        r'\bteleconference\b',  # teleconference
        r'\bconference call\b',  # conference call
        r'\bvideo call\b',  # video call
        r'\bvirtual meeting\b',  # virtual meeting
    ]

    # Communication verbs that strongly indicate interaction reports
    communication_verbs = [
        # German
        r'\bgesprochen mit\b',  # spoke with
        r'\bgesprochen\b',  # spoke
        r'\bgeredet mit\b',  # talked with
        r'\bgeredet\b',  # talked
        r'\bunterhalten mit\b',  # conversed with
        r'\bunterhalten\b',  # conversed
        r'\bdiskutiert mit\b',  # discussed with
        r'\bdiskutiert\b',  # discussed
        r'\bkontaktiert\b',  # contacted
        r'\bin kontakt\b',  # in contact
        r'\bkurzes gespräch\b',  # short conversation

        # English
        r'\btalked\b',  # talked
        r'\btalked to\b',  # talked to
        r'\btalked with\b',  # talked with
        r'\bspoke\b',  # spoke
        r'\bspoke to\b',  # spoke to
        r'\bspoke with\b',  # spoke with
        r'\bdiscussed\b',  # discussed
        r'\bdiscussed with\b',  # discussed with
        r'\bhad a discussion\b',  # had a discussion
        r'\bhad a chat\b',  # had a chat
        r'\bchatted\b',  # chatted
        r'\bchatted with\b',  # chatted with
        r'\bconversed\b',  # conversed
        r'\bconversed with\b',  # conversed with
        r'\bhad a conversation\b',  # had a conversation
    ]

    # Combine call patterns and communication verbs
    call_patterns.extend(communication_verbs)

    visit_patterns = [
        # German visit-related patterns
        r'\bbesuch',  # visit
        r'\bbesucht\b',  # visited
        r'\bvor ort\b',  # on site
        r'\btraf\b',  # met
        r'\bbesuchsbericht\b',  # visit report
        r'\btermin\b',  # appointment
        r'\btermine\b',  # appointments
        r'\bkundentermin\b',  # customer appointment
        r'\bauswärtstermin\b',  # external appointment
        r'\bgespräch vor ort\b',  # on-site meeting
        r'\bgeschäftstermin\b',  # business appointment
        r'\bpersönliches treffen\b',  # personal meeting
        r'\bface-to-face\b',  # face-to-face
        r'\bpersönlich getroffen\b',  # met personally

        # English visit-related patterns
        r'\bvisit',  # visit
        r'\bvisited\b',  # visited
        r'\bmeeting\b',  # meeting
        r'\bon site\b',  # on site
        r'\bmet with\b',  # met with
        r'\bface to face\b',  # face to face
        r'\bin person\b',  # in person
        r'\bon location\b',  # on location
        r'\bsite visit\b',  # site visit
        r'\bcustomer visit\b',  # customer visit
        r'\bfield visit\b',  # field visit
        r'\battended\b',  # attended
        r'\bin attendance\b',  # in attendance
    ]

    # Combined regex for faster execution
    call_regex = re.compile('|'.join(call_patterns), re.IGNORECASE)
    visit_regex = re.compile('|'.join(visit_patterns), re.IGNORECASE)

    # Specific mention patterns that almost always indicate a report
    report_indicators = [
        # German indicators
        r'\bansprechpartner\b',  # contact person
        r'\bgesprächspartner\b',  # conversation partner
        r'\bstandort\b',  # location
        r'\btermin.*?statt',  # appointment took place
        r'\bkontakt',  # contact
        r'\bbericht\b',  # report
        r'\bberichts\b',  # report's
        r'\bnotiz\b',  # note
        r'\bregulär',  # regular
        r'\bgesprächsnotiz\b',  # conversation note
        r'\bkunde\b',  # customer
        r'\bkunden\b',  # customers
        r'\blieferant\b',  # supplier
        r'\bpartner\b',  # partner
        r'\bzusammenarbeit\b',  # collaboration
        r'\bprojekt\b',  # project
        r'\bauftrag\b',  # order/contract
        r'\bangebotsvorstellung\b',  # presentation of offer
        r'\bverhandlung\b',  # negotiation
        r'\babstimmung\b',  # coordination

        # English indicators
        r'\bmeeting.*?with\b',  # meeting with
        r'\bvisit.*?at\b',  # visit at
        r'\bcall.*?with\b',  # call with
        r'\bcontact report\b',  # contact report
        r'\bcontacted\b',  # contacted
        r'\bcustomer\b',  # customer
        r'\bclient\b',  # client
        r'\bsupplier\b',  # supplier
        r'\bvendor\b',  # vendor
        r'\bcontract\b',  # contract
        r'\bproject\b',  # project
        r'\bpresentation\b',  # presentation
        r'\bdiscussion\b',  # discussion
        r'\bnegotiation\b',  # negotiation
        r'\bfollow.?up\b',  # follow-up
        r'\bcheckin\b',  # checkin
        r'\bcheck.?in\b',  # check-in
        r'\bstatus update\b',  # status update
    ]

    report_regex = re.compile('|'.join(report_indicators), re.IGNORECASE)

    # Company/organization indicators
    company_indicators = [
        # German company indicators
        r'\bgmbh\b',  # GmbH (company form)
        r'\bag\b',  # AG (company form)
        r'\bfirma\b',  # company
        r'\bkg\b',  # KG (company form)
        r'\bohg\b',  # OHG (company form)
        r'\bco\.\s?kg\b',  # Co. KG (company form)
        r'\bse\b',  # SE (company form)
        r'\bmbh\b',  # mbH (typo of GmbH)
        r'\bunternehmen\b',  # enterprise
        r'\bkonzern\b',  # corporation
        r'\bgesellschaft\b',  # company/society
        r'\bwerke\b',  # works
        r'\bgruppe\b',  # group
        r'\bholding\b',  # holding

        # English company indicators
        r'\blimited\b',  # Limited (company form)
        r'\binc\b',  # Inc. (company form)
        r'\bco\b',  # Co. (company form)
        r'\bcorp\b',  # Corp. (company form)
        r'\bltd\b',  # Ltd. (company form)
        r'\bllc\b',  # LLC (company form)
        r'\bplc\b',  # PLC (company form)
        r'\bcompany\b',  # company
        r'\bcorporation\b',  # corporation
        r'\benterprise\b',  # enterprise
        r'\bgroup\b',  # group
        r'\bholdings\b',  # holdings
    ]

    company_regex = re.compile('|'.join(company_indicators), re.IGNORECASE)

    # Location + action patterns
    location_action_patterns = [
        # German location+action patterns
        r'(?:war|bin).*?(?:bei|zu besuch|zu gast)',  # was/am at/visiting
        r'(?:komme.*?von|komme.*?aus)',  # coming from
        r'(?:war|bin).*?(?:im|in der)',  # was/am in the
        r'(?:hatte|war).*?(?:termin|besuch)',  # had/was appointment/visit
        r'(?:bin heute|war heute).*?(?:bei|mit)',  # am/was today at/with
        r'(?:habe|hatte).*?(?:vor ort|besucht)',  # have/had on site/visited

        # English location+action patterns
        r'visited.*?(?:at|in)',  # visited at/in
        r'(?:was|am).*?(?:at|in)',  # was/am at/in
        r'(?:just|had).*?(?:meeting|visit)',  # just/had meeting/visit
        r'(?:came|went).*?(?:to|from)',  # came/went to/from
        r'(?:was|had).*?(?:appointment|meeting)',  # was/had appointment/meeting
        r'(?:stopped|visiting).*?(?:by|at)',  # stopped/visiting by/at
    ]

    location_action_regex = re.compile('|'.join(location_action_patterns), re.IGNORECASE)

    # Additional "visit" indicators
    visit_indicator_phrases = [
        # German visit indicators
        r'ich war',  # I was
        r'wir waren',  # we were
        r'(?:heute|gestern|morgen).*?(?:bei|mit)',  # today/yesterday/tomorrow at/with
        r'(?:ich komm|ich komme)',  # I'm coming
        r'(?:ich bin|ich war).*?(?:dort|da)',  # I am/was there
        r'(?:habe|hatte).*?(?:getroffen|besucht)',  # have/had met/visited
        r'(?:gerade|eben).*?(?:bei|von)',  # just now at/from

        # English visit indicators
        r'(?:i was|we were)',  # I was/we were
        r'(?:today|yesterday|tomorrow).*?(?:at|with)',  # today/yesterday/tomorrow at/with
        r'(?:i come|i am coming)',  # I come/I am coming
        r'(?:i am|i was).*?(?:there|here)',  # I am/was there/here
        r'(?:just|recently).*?(?:at|from)',  # just/recently at/from
        r'(?:have|had).*?(?:been to|visited)',  # have/had been to/visited
    ]

    visit_phrases_regex = re.compile('|'.join(visit_indicator_phrases), re.IGNORECASE)

    # Names + titles that often indicate business context
    name_patterns = [
        # German name patterns
        r'(?:herr|frau).*?(?:[A-Z][a-zäöüß]+)',  # Mr./Mrs. followed by capitalized name
        r'(?:dr\.|prof\.).*?(?:[A-Z][a-zäöüß]+)',  # Dr./Prof. followed by capitalized name
        r'(?:dipl\.|ing\.).*?(?:[A-Z][a-zäöüß]+)',  # Dipl./Ing. followed by capitalized name

        # English name patterns
        r'(?:mr\.|mrs\.|ms\.).*?(?:[A-Z][a-z]+)',  # Mr./Mrs./Ms. followed by capitalized name
        r'(?:dr\.|prof\.).*?(?:[A-Z][a-z]+)',  # Dr./Prof. followed by capitalized name
        r'(?:sir|madam).*?(?:[A-Z][a-z]+)',  # Sir/Madam followed by capitalized name
    ]

    name_regex = re.compile('|'.join(name_patterns), re.IGNORECASE)

    # Job titles that indicate business context
    job_title_patterns = [
        # German job titles
        r'\bgeschäftsführer\b',  # CEO/Managing Director
        r'\bvorstand\b',  # board member
        r'\babteilungsleiter\b',  # department head
        r'\bleiter\b',  # head/manager
        r'\bmanager\b',  # manager
        r'\beinkäufer\b',  # purchaser
        r'\bverkaufsleiter\b',  # sales manager
        r'\bvertriebsleiter\b',  # sales director

        # English job titles
        r'\bceo\b',  # CEO
        r'\bcfo\b',  # CFO
        r'\bcto\b',  # CTO
        r'\bmanager\b',  # manager
        r'\bdirector\b',  # director
        r'\bhead of\b',  # head of
        r'\bvp\b',  # VP
        r'\bexecutive\b',  # executive
        r'\bsupervisor\b',  # supervisor
        r'\bteam lead\b',  # team lead
    ]

    job_title_regex = re.compile('|'.join(job_title_patterns), re.IGNORECASE)

    # Time expressions often found in meeting/call reports
    time_patterns = [
        # German time expressions
        r'\bheute morgen\b',  # this morning
        r'\bheute nachmittag\b',  # this afternoon
        r'\bgestern\b',  # yesterday
        r'\bletzte woche\b',  # last week
        r'\bnächste woche\b',  # next week
        r'\bvon \d{1,2}:\d{2}\b',  # from HH:MM
        r'\bbis \d{1,2}:\d{2}\b',  # until HH:MM
        r'\b\d{1,2}:\d{2} uhr\b',  # HH:MM o'clock

        # English time expressions
        r'\bthis morning\b',  # this morning
        r'\bthis afternoon\b',  # this afternoon
        r'\byesterday\b',  # yesterday
        r'\blast week\b',  # last week
        r'\bnext week\b',  # next week
        r'\bfrom \d{1,2}:\d{2}\b',  # from HH:MM
        r'\buntil \d{1,2}:\d{2}\b',  # until HH:MM
        r'\bat \d{1,2}:\d{2}\b',  # at HH:MM
    ]

    time_regex = re.compile('|'.join(time_patterns), re.IGNORECASE)

    dirty_indices = []

    for i, text in enumerate(texts):
        if text is None or not isinstance(text, str):
            continue

        # Skip very short texts
        if len(text.split()) < 3:
            continue

        # Strong direct indicators - automatically classify as dirty
        if (
                # Strong report-specific phrases
                re.search(r'\bbesuchsbericht\b|\btelefonbericht\b|\bcontact report\b|\bvisit report\b|\bcall report\b',
                          text, re.IGNORECASE) or
                # Opening phrases typically used in reports
                re.search(r'^(?:telefonat mit|besuch bei|gespräch mit|meeting with|call with|visited)', text,
                          re.IGNORECASE) or
                # Meeting duration formats
                re.search(r'meeting.*?from.*?to|meeting.*?lasted|termin.*?von.*?bis', text, re.IGNORECASE)
        ):
            dirty_indices.append(i)
            continue

        # Check if this is likely a call or visit report
        is_call_related = call_regex.search(text) is not None
        is_visit_related = visit_regex.search(text) is not None

        if is_call_related or is_visit_related:
            # Look for supporting evidence to confirm it's a report
            has_supporting_evidence = (
                    report_regex.search(text) is not None or
                    company_regex.search(text) is not None or
                    location_action_regex.search(text) is not None or
                    name_regex.search(text) is not None or
                    job_title_regex.search(text) is not None or
                    time_regex.search(text) is not None
            )

            if has_supporting_evidence:
                dirty_indices.append(i)
                continue

        # Check for visit phrases and other indicators even without
        # explicit "call" or "visit" mentions
        if visit_phrases_regex.search(text):
            # Look for business context indicators
            has_business_context = (
                    company_regex.search(text) is not None or
                    name_regex.search(text) is not None or
                    job_title_regex.search(text) is not None or
                    report_regex.search(text) is not None
            )

            if has_business_context:
                dirty_indices.append(i)
                continue

        # Look for initial phrases that often start reports
        if re.search(r'^(?:war|besuch|meeting|call|telefonat|gespräch|visit)', text, re.IGNORECASE):
            # Check if followed by company or person
            if company_regex.search(text) or name_regex.search(text):
                dirty_indices.append(i)

    print(f"Pattern matching identified {len(dirty_indices)} potential dirty samples")
    return dirty_indices

def refine_with_ml(texts, initial_dirty_indices):
    """
    Use a simple ML approach to refine the classification

    Args:
        texts: NumPy array of text data
        initial_dirty_indices: Initial dirty indices from pattern matching

    Returns:
        refined_dirty_indices: Refined list of dirty indices
    """
    # Create initial labels based on pattern matching
    labels = np.zeros(len(texts), dtype=int)
    labels[initial_dirty_indices] = 1

    # Create a simple bag-of-words model
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=2,
        stop_words='english',  # Will help with English text
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )

    # Transform the text data
    X = vectorizer.fit_transform([t if isinstance(t, str) else "" for t in texts])

    # Train a simple Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, labels)

    # Get the refined predictions
    predicted_labels = classifier.predict(X)

    # Get the new dirty indices
    refined_dirty_indices = np.where(predicted_labels == 1)[0].tolist()

    # Print the classification report
    print("Classification Report:")
    print(classification_report(labels, predicted_labels))

    print(f"ML refinement identified {len(refined_dirty_indices)} dirty samples")
    return refined_dirty_indices


def write_array_to_txt(npy_path, txt_path):
    """Write NumPy array contents to a text file, one element per line."""
    try:
        data = np.load(npy_path, allow_pickle=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for item in data:
                if item is not None:
                    f.write(f"{item}\n")
        print(f"Wrote {len(data)} items to {txt_path}")
    except Exception as e:
        print(f"Error writing {txt_path}: {e}")


def classify_and_save(texts_path, embeddings_path, output_dir=None, save_with_suffix="_improved"):
    """
    Main function to classify texts and save the results

    Args:
        texts_path: Path to the text data file (.npy)
        embeddings_path: Path to the embeddings file (.npy)
        output_dir: Directory to save output files (if None, saves in same directory as input)
        save_with_suffix: Suffix to add to the output filenames

    Returns:
        Dictionary with paths to saved files and count information
    """
    # Load the data
    texts, embeddings = load_data(texts_path, embeddings_path)
    if texts is None or embeddings is None:
        print("Failed to load data")
        return

    # Get initial classification with pattern matching
    dirty_indices = identify_call_visit_patterns(texts)

    # Refine with ML approach
    refined_dirty_indices = refine_with_ml(texts, dirty_indices)

    # Use the refined indices
    final_dirty_indices = sorted(refined_dirty_indices)

    # Create boolean mask for clean (not dirty) indices
    mask = np.ones(len(texts), dtype=bool)
    mask[final_dirty_indices] = False

    # Get the clean indices
    final_clean_indices = np.where(mask)[0].tolist()

    print(f"Final classification: {len(final_clean_indices)} clean samples, {len(final_dirty_indices)} dirty samples")

    # Report the ratio but don't try to enforce any specific split
    dirty_ratio = len(final_dirty_indices) / len(texts)
    print(f"Dirty ratio: {dirty_ratio:.2%}")

    # Just report the ratio - no adjustment needed as we want the natural distribution
    clean_ratio = len(final_clean_indices) / len(texts)
    print(f"Final distribution: {clean_ratio:.2%} clean, {dirty_ratio:.2%} dirty")

    # Create and save the clean and dirty datasets
    clean_texts = texts[mask]
    clean_embeddings = embeddings[mask]

    dirty_texts = texts[~mask]
    dirty_embeddings = embeddings[~mask]

    # Determine output directories
    if output_dir is None:
        # Use the directories of the input files
        text_dir = os.path.dirname(texts_path)
        embeddings_dir = os.path.dirname(embeddings_path)
    else:
        # Use the provided output directory
        text_dir = os.path.join(output_dir, "texts")
        embeddings_dir = os.path.join(output_dir, "embeddings")

    # Create output directories if they don't exist
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Get base filenames
    text_base = os.path.basename(texts_path).replace(".npy", "")
    emb_base = os.path.basename(embeddings_path).replace(".npy", "")

    # Save clean data
    clean_texts_path = os.path.join(text_dir, f"{text_base}_clean{save_with_suffix}.npy")
    clean_embeddings_path = os.path.join(embeddings_dir, f"{emb_base}_clean{save_with_suffix}.npy")

    # Save dirty data
    dirty_texts_path = os.path.join(text_dir, f"{text_base}_dirty{save_with_suffix}.npy")
    dirty_embeddings_path = os.path.join(embeddings_dir, f"{emb_base}_dirty{save_with_suffix}.npy")

    # Save the arrays
    np.save(clean_texts_path, clean_texts)
    np.save(clean_embeddings_path, clean_embeddings)
    np.save(dirty_texts_path, dirty_texts)
    np.save(dirty_embeddings_path, dirty_embeddings)

    # Also save as text files for easy inspection
    write_array_to_txt(clean_texts_path, clean_texts_path.replace('.npy', '.txt'))
    write_array_to_txt(dirty_texts_path, dirty_texts_path.replace('.npy', '.txt'))

    print(f"Saved {len(clean_texts)} clean samples and {len(dirty_texts)} dirty samples")
    print(f"Clean texts saved to: {clean_texts_path}")
    print(f"Clean embeddings saved to: {clean_embeddings_path}")
    print(f"Dirty texts saved to: {dirty_texts_path}")
    print(f"Dirty embeddings saved to: {dirty_embeddings_path}")

    return {
        'clean_texts_path': clean_texts_path,
        'clean_embeddings_path': clean_embeddings_path,
        'dirty_texts_path': dirty_texts_path,
        'dirty_embeddings_path': dirty_embeddings_path,
        'clean_count': len(clean_texts),
        'dirty_count': len(dirty_texts)
    }


if __name__ == "__main__":
    # Example of how to use the classifier with explicit paths
    parser = argparse.ArgumentParser(description="Classify text data into clean and dirty categories")
    parser.add_argument("--texts", type=str, required=True, help="Path to text data (.npy)")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings data (.npy)")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files (default: same as input)")
    parser.add_argument("--suffix", type=str, default="_improved", help="Suffix to add to output filenames")

    args = parser.parse_args()

    # Run the classification
    result = classify_and_save(
        texts_path=args.texts,
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        save_with_suffix=args.suffix
    )

    # Print summary
    if result:
        print("\nClassification Summary:")
        print(f"Total samples: {result['clean_count'] + result['dirty_count']}")
        print(f"Clean samples: {result['clean_count']}")
        print(f"Dirty samples: {result['dirty_count']}")
        print(f"Clean ratio: {result['clean_count'] / (result['clean_count'] + result['dirty_count']):.2%}")
        print(f"Dirty ratio: {result['dirty_count'] / (result['clean_count'] + result['dirty_count']):.2%}")