import json
import csv
import random
from collections import defaultdict

class TagsTokenizer:
    """
    A standalone class to handle tokenization of music features.

    This class loads all necessary data upon initialization and provides a
    single method, get_tokens, to generate an 8-token list for a given
    track ID, artist ID, or feature dictionary.
    """
    def __init__(self, tsv_path='/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/data/autotagging.tsv', mappings_path='/home/tzlillev/ProjectAudioMTG/datasets_music/token_mappings.json'):
        """
        Initializes the tokenizer by loading all required data.
        
        Args:
            tsv_path (str): Path to the autotagging.tsv file.
            mappings_path (str): Path to the token_mappings.json file.
        """
        print("Initializing MusicTokenizer...")
        self.padding_token = 2050
        
        print("Loading token mappings...")
        self.mappings = self._load_token_mappings(mappings_path)
        
        print("Parsing track and artist data from TSV...")
        self.track_data, self.artist_data = self._parse_data(tsv_path)
        
        if self.mappings and self.track_data:
            print("\n✅ Tokenizer ready.\n" + "-"*25)
        else:
            print("\n❌ Tokenizer initialization failed. Please check file paths.")

    def _load_token_mappings(self, file_path):
        """Loads the pre-generated token mappings from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Mapping file not found at '{file_path}'.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}'.")
            return None

    def _parse_data(self, file_path):
        """
        Parses the TSV file to create dictionaries for tracks and artists.
        This version correctly handles rows with a variable number of tags.
        """
        tracks = {}
        artist_features_agg = defaultdict(lambda: {"mood": set(), "genre": set(), "instruments": set()})
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
                # Use csv.reader for robust handling of variable-length rows
                reader = csv.reader(tsvfile, delimiter='\t')
                # Skip the header row
                header = next(reader)
                
                for row in reader:
                    # Ensure the row has at least the minimum number of columns
                    if len(row) < 6:
                        continue

                    track_id = row[0]
                    artist_id = row[1]
                    
                    # All items from the 6th column onwards are considered tags
                    all_tags = row[5:]
                    
                    track_features = {"mood": [], "genre": [], "instruments": []}
                    for tag in all_tags:
                        tag = tag.strip()
                        if 'genre---' in tag:
                            feature = tag.split('---', 1)[1]
                            track_features["genre"].append(feature)
                            artist_features_agg[artist_id]["genre"].add(feature)
                        elif 'mood/theme---' in tag:
                            feature = tag.split('---', 1)[1]
                            track_features["mood"].append(feature)
                            artist_features_agg[artist_id]["mood"].add(feature)
                        elif 'instrument---' in tag:
                            feature = tag.split('---', 1)[1]
                            track_features["instruments"].append(feature)
                            artist_features_agg[artist_id]["instruments"].add(feature)
                    tracks[track_id] = track_features
        except FileNotFoundError:
            print(f"Error: Track data file not found at '{file_path}'.")
            return None, None
            
        artists = {
            artist_id: {
                "mood": sorted(list(feats["mood"])),
                "genre": sorted(list(feats["genre"])),
                "instruments": sorted(list(feats["instruments"])),
            }
            for artist_id, feats in artist_features_agg.items()
        }
        
        return tracks, artists

    def get_tokens(self, input_data):
        """
        Generates an 8-token list based on input features, a track ID, or an artist ID.
        """
        if not self.mappings or not self.track_data:
            return {"error": "Tokenizer is not properly initialized."}

        features_to_tokenize = {}
        input_type = "Dictionary"

        if isinstance(input_data, int):
            input_type = "Track ID"
            track_id_str = f"track_{input_data:07d}"
            features_to_tokenize = self.track_data.get(track_id_str, {}).copy()
            if not features_to_tokenize:
                return {"error": f"Track ID '{track_id_str}' not found."}

        elif isinstance(input_data, str) and input_data.startswith('artist_'):
            input_type = "Artist ID"
            features_to_tokenize = self.artist_data.get(input_data, {}).copy()
            if not features_to_tokenize:
                return {"error": f"Artist ID '{input_data}' not found."}
                
        elif isinstance(input_data, dict):
            features_to_tokenize = input_data
        else:
            return {"error": "Invalid input. Provide an integer track ID, an artist ID string, or a dictionary."}

        if input_type in ["Track ID", "Artist ID"]:
            if len(features_to_tokenize.get("genre", [])) > 2:
                features_to_tokenize["genre"] = random.sample(features_to_tokenize["genre"], 2)
            if len(features_to_tokenize.get("mood", [])) > 2:
                features_to_tokenize["mood"] = random.sample(features_to_tokenize["mood"], 2)
            if len(features_to_tokenize.get("instruments", [])) > 4:
                features_to_tokenize["instruments"] = random.sample(features_to_tokenize["instruments"], 4)

        genre_map = self.mappings.get("genre_map", {})
        mood_map = self.mappings.get("mood_map", {})
        instrument_map = self.mappings.get("instrument_map", {})

        genre_tokens = [genre_map.get(g, self.padding_token) for g in features_to_tokenize.get("genre", [])]
        mood_tokens = [mood_map.get(m, self.padding_token) for m in features_to_tokenize.get("mood", [])]
        instrument_tokens = [instrument_map.get(i, self.padding_token) for i in features_to_tokenize.get("instruments", [])]

        padded_genres = (genre_tokens + [self.padding_token] * 2)[:2]
        padded_moods = (mood_tokens + [self.padding_token] * 2)[:2]
        padded_instruments = (instrument_tokens + [self.padding_token] * 4)[:4]

        final_tokens = padded_genres + padded_moods + padded_instruments
        
        return {
            "tokens": final_tokens,
            "features_used": features_to_tokenize
        }

def test_tokenizer():
    """Test the tokenizer with various inputs and verify outputs."""
    # Initialize the Tags Tokenizer
    tokenizer = TagsTokenizer()

    # Test with the specific track ID that was failing
    track_result = tokenizer.get_tokens(1041563)
    print(f"\n--- Testing Track ID 1041563 ---")
    print(f"Result: {track_result}")
    assert isinstance(track_result, dict), "Track result should be a dictionary"
    assert "tokens" in track_result, "Track result should contain tokens"
    assert len(track_result["tokens"]) == 8, "Should have 8 tokens (2 genre + 2 mood + 4 instruments)"
    # Check that more than one token is not the padding token
    assert track_result["tokens"].count(2050) < 7, "Should have more than one valid token for this track"
    print("✅ Track ID test passed.")
    
    # Test with dictionary input
    test_dict = {
        "genre": ["metal", "rock"],
        "mood": ["dark", "powerful"],
        "instruments": ["electricguitar", "drums", "bass"]
    }

    test_dict = {"genre": ["raggae"], "mood": ["happy"]}
    dict_result = tokenizer.get_tokens(test_dict)
    print(f"\n--- Testing Dictionary Input ---")
    print(f"Result: {dict_result} for {test_dict}")
    assert isinstance(dict_result, dict), "Dictionary result should be a dictionary"
    assert "tokens" in dict_result, "Dictionary result should contain tokens"
    assert len(dict_result["tokens"]) == 8, "Should have 8 tokens"
    print("✅ Dictionary test passed.")

    # Test error cases
    invalid_result = tokenizer.get_tokens("invalid_input")
    assert "error" in invalid_result, "Should return error for invalid input"
    print("\n✅ Error case test passed.")

    print("\nAll tokenizer tests passed!")
    return True

# This will now run your test function when the script is executed
if __name__ == '__main__':
    test_tokenizer()

# import json
# import json
# import csv
# import random

# def load_token_mappings(file_path='/home/tzlillev/ProjectAudioMTG/datasets_music/token_mappings.json'):
#     """Loads the pre-generated token mappings from a JSON file."""
#     try:
#         with open(file_path, 'r') as f:
#             mappings = json.load(f)
#         return mappings
#     except FileNotFoundError:
#         print(f"Error: Mapping file not found at '{file_path}'.")
#         return None
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from '{file_path}'.")
#         return None

# def parse_track_data(file_path='/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/data/autotagging.tsv'):
#     """
#     Parses the TSV file to create a dictionary mapping each track_id
#     to its list of features for each category.
#     """
#     tracks = {}
#     try:
#         with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
#             reader = csv.DictReader(tsvfile, delimiter='\t')
#             for row in reader:
#                 track_id = row.get('TRACK_ID') or row.get('track_id')
#                 tags_str = row.get('TAGS') or row.get('tags')
#                 if not track_id or not tags_str:
#                     continue

#                 track_features = {"mood": [], "genre": [], "instruments": []}
#                 for tag in tags_str.split('\t'):
#                     tag = tag.strip()
#                     if 'genre---' in tag:
#                         track_features["genre"].append(tag.split('---', 1)[1])
#                     elif 'mood/theme---' in tag:
#                         track_features["mood"].append(tag.split('---', 1)[1])
#                     elif 'instrument---' in tag:
#                         track_features["instruments"].append(tag.split('---', 1)[1])
#                 tracks[track_id] = track_features
#     except FileNotFoundError:
#         print(f"Error: Track data file not found at '{file_path}'.")
#         return None
#     return tracks

# def get_tokens(input_data, mappings, track_data):
#     """
#     Generates an 8-token list based on input features or a track ID.

#     Args:
#         input_data: An integer track ID or a dictionary of features.
#         mappings (dict): The loaded token mappings.
#         track_data (dict): The parsed data from the TSV file.

#     Returns:
#         dict: A dictionary containing the final tokens and the features used.
#     """
#     padding_token = 2050
#     features_to_tokenize = {}

#     if isinstance(input_data, int):
#         track_id_str = f"track_{input_data:07d}"
#         original_features = track_data.get(track_id_str)
#         if not original_features:
#             return {"error": f"Track ID '{track_id_str}' not found."}
        
#         # Make a copy to avoid modifying the original data
#         features_to_tokenize = original_features.copy()

#         # --- Randomly sample if features exceed limits ---
#         if len(features_to_tokenize.get("genre", [])) > 2:
#             features_to_tokenize["genre"] = random.sample(features_to_tokenize["genre"], 2)
        
#         if len(features_to_tokenize.get("mood", [])) > 2:
#             features_to_tokenize["mood"] = random.sample(features_to_tokenize["mood"], 2)

#         if len(features_to_tokenize.get("instruments", [])) > 4:
#             features_to_tokenize["instruments"] = random.sample(features_to_tokenize["instruments"], 4)

#     elif isinstance(input_data, dict):
#         features_to_tokenize = input_data
#     else:
#         return {"error": "Invalid input. Provide an integer track ID or a dictionary."}

#     # --- Tokenize the features ---
#     genre_map = mappings.get("genre_map", {})
#     mood_map = mappings.get("mood_map", {})
#     instrument_map = mappings.get("instrument_map", {})

#     genre_tokens = [genre_map.get(g, padding_token) for g in features_to_tokenize.get("genre", [])]
#     mood_tokens = [mood_map.get(m, padding_token) for m in features_to_tokenize.get("mood", [])]
#     instrument_tokens = [instrument_map.get(i, padding_token) for i in features_to_tokenize.get("instruments", [])]

#     # --- Pad to the correct length ---
#     padded_genres = (genre_tokens + [padding_token] * 2)[:2]
#     padded_moods = (mood_tokens + [padding_token] * 2)[:2]
#     padded_instruments = (instrument_tokens + [padding_token] * 4)[:4]

#     final_tokens = padded_genres + padded_moods + padded_instruments
    
#     return {
#         "tokens": final_tokens,
#         "features_used": features_to_tokenize
#     }

# def main():
#     """
#     Main function to load all data and demonstrate the get_tokens function.
#     """
#     # --- Load all necessary data first ---
#     print("Loading token mappings...")
#     mappings = load_token_mappings()
#     if not mappings: return

#     print("Parsing track data from TSV...")
#     track_data = parse_track_data()
#     if not track_data: return
    
#     print("\n✅ All data loaded successfully.\n" + "-"*25)

#     # --- Example 1: Query with a dictionary ---
#     dict_input = {
#         "genre": ["pop", "rock"],
#         "mood": ["happy"],
#         "instruments": ["guitar", "drums", "bass", "piano", "violin"]
#     }
#     print(f"🔎 Querying with a dictionary:\n{dict_input}")
#     result1 = get_tokens(dict_input, mappings, track_data)
#     print(f"   Tokens: {result1['tokens']}")
#     print(f"   Features Used: {result1['features_used']}")
#     print("-" * 25)

#     # --- Example 2: Query with a valid track ID ---
#     # We'll use a known track ID from the dataset
#     track_id_input = 215 # Corresponds to track_0000215 which has genre 'metal'
#     print(f"🔎 Querying with Track ID: {track_id_input}")
#     result2 = get_tokens(track_id_input, mappings, track_data)
#     # For track_id, we only print the tags for debug purposes as requested
#     print(f"   Tokens: {result2['tokens']}")
#     print(f"   DEBUG - Features Found: {result2['features_used']}")
#     print("-" * 25)

#     # --- Example 3: Find and query a track with > 4 instruments to test sampling ---
#     track_to_test = None
#     for tid, feats in track_data.items():
#         if len(feats.get('instruments', [])) > 4:
#             track_to_test = tid
#             break
    
#     if track_to_test:
#         track_id_int = int(track_to_test.split('_')[-1])
#         print(f"🔎 Testing random sampling with Track ID: {track_id_int}")
#         print(f"   Original Instruments ({len(track_data[track_to_test]['instruments'])}): {track_data[track_to_test]['instruments']}")
#         result3 = get_tokens(track_id_int, mappings, track_data)
#         print(f"   Tokens: {result3['tokens']}")
#         print(f"   Features Used (Sampled down to 4): {result3['features_used']['instruments']}")
#     else:
#         print("Could not find a track with > 4 instruments to test sampling.")
#     print("-" * 25)


# if __name__ == '__main__':
#     main()





# def create_token_mappings(genres, moods, instruments):
#     """
#     Creates separate mapping dictionaries for each category, mapping each
#     feature to a unique integer ID.

#     Args:
#         genres (list): A list of unique genre strings.
#         moods (list): A list of unique mood strings.
#         instruments (list): A list of unique instrument strings.

#     Returns:
#         tuple: A tuple containing three dictionaries: 
#                (genre_map, mood_map, instrument_map).
#     """
#     # The padding token can be used for empty slots later.
#     padding_token = 2050
    
#     # Create a mapping by enumerating the sorted list of features.
#     # We start IDs from 1, so 0 can be reserved if needed.
#     genre_map = {feature: i + 1 for i, feature in enumerate(sorted(genres))}
#     mood_map = {feature: i + 1 for i, feature in enumerate(sorted(moods))}
#     instrument_map = {feature: i + 1 for i, feature in enumerate(sorted(instruments))}
    
#     # Optionally, you could add the padding token to the map,
#     # but it's often handled separately. For now, we'll return the pure maps.

#     return genre_map, mood_map, instrument_map

# # def main():
# #     """
# #     Main function to demonstrate the mapping creation and save the result.
# #     """
# #     # --- Paste the lists from the previous step ---
# #     unique_genres = ['60s', '70s', '80s', '90s', 'acidjazz', 'african', 'alternative', 'alternativerock', 'ambient', 'atmospheric', 'blues', 'bluesrock', 'bossanova', 'breakbeat', 'celtic', 'chanson', 'chillout', 'choir', 'classical', 'classicrock', 'club', 'contemporary', 'country', 'dance', 'darkambient', 'darkwave', 'deephouse', 'disco', 'downtempo', 'drumnbass', 'dub', 'dubstep', 'easylistening', 'edm', 'electronic', 'electronica', 'electropop', 'ethnicrock', 'ethno', 'eurodance', 'experimental', 'folk', 'funk', 'fusion', 'gothic', 'groove', 'grunge', 'hard', 'hardrock', 'heavymetal', 'hiphop', 'house', 'idm', 'improvisation', 'indie', 'industrial', 'instrumentalpop', 'instrumentalrock', 'jazz', 'jazzfunk', 'jazzfusion', 'latin', 'lounge', 'medieval', 'metal', 'minimal', 'newage', 'newwave', 'orchestral', 'oriental', 'pop', 'popfolk', 'poprock', 'postrock', 'progressive', 'psychedelic', 'punkrock', 'rap', 'reggae', 'rnb', 'rock', 'rocknroll', 'singersongwriter', 'ska', 'soul', 'soundtrack', 'swing', 'symphonic', 'synthpop', 'techno', 'trance', 'tribal', 'triphop', 'world']
# #     unique_moods = ['adventure', 'advertising', 'children', 'christmas', 'commercial', 'corporate', 'dark', 'dramatic', 'dream', 'energetic', 'fast', 'film', 'game', 'happy', 'holiday', 'love', 'meditative', 'melodic', 'party', 'powerful', 'relaxing', 'retro', 'sexy', 'soundscape', 'space']
# #     unique_instruments = ['accordion', 'acousticbassguitar', 'acousticguitar', 'bass', 'bell', 'brass', 'cello', 'clarinet', 'computer', 'doublebass', 'drummachine', 'drums', 'electricguitar', 'electricpiano', 'flute', 'guitar', 'orchestra', 'percussion', 'piano', 'sampler', 'synthesizer', 'ukulele', 'viola', 'violin']

# #     # --- Create the mappings ---
# #     genre_map, mood_map, instrument_map = create_token_mappings(
# #         unique_genres, unique_moods, unique_instruments
# #     )

# #     # --- Combine into a single object for saving ---
# #     full_mapping = {
# #         "genre_map": genre_map,
# #         "mood_map": mood_map,
# #         "instrument_map": instrument_map
# #     }

# #     # --- Print the results and save to a file ---
# #     output_filename = '/home/tzlillev/ProjectAudioMTG/datasets_music/token_mappings.json'
# #     print(f"--- Genre Mapping (first 5) ---")
# #     print({k: genre_map[k] for k in list(genre_map)[:5]})
    
# #     print(f"\n--- Mood Mapping (first 5) ---")
# #     print({k: mood_map[k] for k in list(mood_map)[:5]})

# #     print(f"\n--- Instrument Mapping (first 5) ---")
# #     print({k: instrument_map[k] for k in list(instrument_map)[:5]})

# #     try:
# #         with open(output_filename, 'w') as f:
# #             json.dump(full_mapping, f, indent=4)
# #         print(f"\n✅ Successfully saved all mappings to '{output_filename}'")
# #     except Exception as e:
# #         print(f"\n❌ Error saving file: {e}")


# # if __name__ == '__main__':
# #     main()