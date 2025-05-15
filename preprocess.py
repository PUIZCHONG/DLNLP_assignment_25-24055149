import os
import re
import ast
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

class NBMEDataProcessor:
    """
    NBME clinical patient notes data processor, integrating the best data processing practices from two notebooks
    """
    def __init__(self, data_dir=r"C:\Users\SIMON\Desktop\NLP\nbme-score-clinical-patient-notes", output_dir=r"C:\Users\SIMON\Desktop\NLP\processed"):
        """
        Initialize NBME data processor
        
        Args:
            data_dir: Input data directory
            output_dir: Output directory for processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train = None
        self.test = None
        self.patient_notes = None
        self.features = None
        self.train_processed = None
        self.final_data = None
        self.n_folds = 5  # Default number of folds
        
        # Feature category grouping
        self.feature_female = []  # List of female-related feature numbers
        self.feature_male = []    # List of male-related feature numbers
        self.feature_year = []    # List of age-related feature numbers
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Ignore warnings
        warnings.filterwarnings("ignore")
        
        # Medical abbreviation dictionary (adopted from the first notebook)
        self.medical_abbreviations = {
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'chf': 'congestive heart failure',
            'cad': 'coronary artery disease',
            'mi': 'myocardial infarction',
            'afib': 'atrial fibrillation',
            'copd': 'chronic obstructive pulmonary disease',
            'uti': 'urinary tract infection',
            'bph': 'benign prostatic hyperplasia',
            'gerd': 'gastroesophageal reflux disease',
            'hx': 'history',
            'yo': 'year old',
            'y/o': 'year old',
            'yo/': 'year old',
            'y.o.': 'year old',
            'w/': 'with',
            's/p': 'status post',
            'h/o': 'history of',
            'c/o': 'complains of',
            'p/w': 'presenting with',
            'neg': 'negative',
            'pos': 'positive',
            '+': 'positive',
            '-': 'negative',
            'w/o': 'without',
            'b/l': 'bilateral',
            'r/o': 'rule out',
            '&': 'and',
            'pt': 'patient',
            'sx': 'symptoms',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'fx': 'fracture',
            'vs': 'vital signs',
        }
    
    def load_data(self):
        """Load all necessary data files"""
        try:
            print("Loading data files...")
            self.train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            self.patient_notes = pd.read_csv(os.path.join(self.data_dir, 'patient_notes.csv'))
            self.features = pd.read_csv(os.path.join(self.data_dir, 'features.csv'))
            
            # Load test data (if exists)
            test_path = os.path.join(self.data_dir, 'test.csv')
            if os.path.exists(test_path):
                self.test = pd.read_csv(test_path)
                
            print(f"Loaded train data with {len(self.train)} rows")
            print(f"Loaded patient notes with {len(self.patient_notes)} rows")
            print(f"Loaded features with {len(self.features)} rows")
            if self.test is not None:
                print(f"Loaded test data with {len(self.test)} rows")
                
            # Identify feature types
            self.identify_feature_types()
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def identify_feature_types(self):
        """Identify different types of features (gender, age, etc.)"""
        print("Identifying feature types...")
        
        # Reset feature type lists
        self.feature_female = []
        self.feature_male = []
        self.feature_year = []
        
        # Iterate through features
        for idx, row in self.features.iterrows():
            feature_text = row['feature_text'].lower()
            feature_num = row['feature_num']
            
            # Identify female-related features
            if any(term in feature_text for term in ['female', 'woman', 'girl', 'mother', 'sister', 'daughter']):
                self.feature_female.append(feature_num)
                
            # Identify male-related features
            if any(term in feature_text for term in ['male', 'man', 'boy', 'father', 'brother', 'son']):
                self.feature_male.append(feature_num)
                
            # Identify age-related features
            if any(term in feature_text for term in ['age', 'year old', 'y.o', 'yo', 'y/o']):
                self.feature_year.append(feature_num)
                
        print(f"Identified {len(self.feature_female)} female-related features")
        print(f"Identified {len(self.feature_male)} male-related features")
        print(f"Identified {len(self.feature_year)} age-related features")
    
    def preprocess_features(self):
        """Process special cases in feature text (from the second notebook)"""
        # Fix the text of the 27th feature
        self.features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
        # More feature preprocessing logic can be added
        return self.features
    
    def parse_annotations(self):
        """Convert string format annotations and locations to lists"""
        # Ensure annotations and locations are parsed into Python objects
        if isinstance(self.train['annotation'].iloc[0], str):
            self.train['annotation'] = self.train['annotation'].apply(ast.literal_eval)
        
        if isinstance(self.train['location'].iloc[0], str):
            self.train['location'] = self.train['location'].apply(ast.literal_eval)
            
        # Add annotation length field
        self.train['annotation_length'] = self.train['annotation'].apply(len)
        return self.train
    
    def merge_data(self):
        """Merge training data with features and patient notes"""
        if self.train is None or self.features is None or self.patient_notes is None:
            print("Please load data first.")
            return None
        
        # Merge data
        self.train = self.train.merge(self.features, on=['feature_num', 'case_num'], how='left')
        self.train = self.train.merge(self.patient_notes, on=['pn_num', 'case_num'], how='left')
        return self.train
    
    def check_annotation_integrity(self):
        """
        Check the completeness and consistency of annotations
        instead of manually correcting specific errors
        """
        print("Checking annotation integrity...")
        
        # Create a copy to avoid modifying the original data
        checked_train = self.train.copy()
        
        # Check rows with empty annotations but location information
        empty_annot_with_loc = checked_train[
            (checked_train['annotation_length'] == 0) & 
            (checked_train['location'].apply(lambda x: len(x) > 0))
        ]
        
        if len(empty_annot_with_loc) > 0:
            print(f"Warning: Found {len(empty_annot_with_loc)} rows with empty annotations but location data")
            
        # Check rows with annotations but no location information
        annot_without_loc = checked_train[
            (checked_train['annotation_length'] > 0) & 
            (checked_train['location'].apply(lambda x: len(x) == 0))
        ]
        
        if len(annot_without_loc) > 0:
            print(f"Warning: Found {len(annot_without_loc)} rows with annotations but no location data")
        
        # More integrity checks can be added...
        
        return checked_train
    
    def standardize_medical_text(self):
        """
        Standardize medical terminology (adopted and optimized from the first notebook)
        """
        print("Standardizing medical text...")
        
        # Create a working copy
        train_standardized = self.train.copy()
        
        # Standardized text processing function
        def standardize_text(text):
            if pd.isna(text):
                return text
                
            # Replace medical abbreviations
            for abbr, full_form in self.medical_abbreviations.items():
                pattern = r'\b' + re.escape(abbr) + r'\b'
                text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
                
            return text
        
        # Standardize patient history text
        train_standardized['pn_history'] = train_standardized['pn_history'].apply(standardize_text)
        
        # Standardize feature text
        train_standardized['feature_text'] = train_standardized['feature_text'].apply(standardize_text)
        
        self.train_standardized = train_standardized
        print("Medical text standardization completed")
        return self.train_standardized
    
    def correct_offsets(self):
        """
        Correct annotation positions after text standardization (optimized from the first notebook)
        """
        print("Correcting annotation offsets...")
        
        if not hasattr(self, 'train_standardized'):
            print("Standardized data not found. Running standardize_medical_text first...")
            self.standardize_medical_text()
        
        # Create a working copy
        train_offset_corrected = self.train_standardized.copy()
        
        # Since text standardization may have changed the text length, positions need to be updated
        # This implementation is simplified; actual application requires more complex logic
        
        def adjust_location(row):
            """Adjust position offsets"""
            if not row['location'] or pd.isna(row['pn_history']):
                return row['location']
                
            # Get the original patient note text
            original_text = self.train.loc[row.name, 'pn_history']
            
            # Get the standardized patient note text
            standardized_text = row['pn_history']
            
            adjusted_locations = []
            for loc_list in row['location']:
                adjusted_loc_parts = []
                
                for loc in loc_list.split(';'):
                    if ' ' in loc:
                        start, end = map(int, loc.split())
                        # Extract the phrase from the original text
                        if start < len(original_text) and end <= len(original_text):
                            phrase = original_text[start:end]
                            
                            # Find the phrase in the standardized text
                            # Note: This is a simplified method, a more complex method may be needed to handle multiple occurrences
                            if phrase in standardized_text:
                                new_start = standardized_text.find(phrase)
                                new_end = new_start + len(phrase)
                                adjusted_loc_parts.append(f"{new_start} {new_end}")
                            else:
                                # If an exact match cannot be found, use the original position
                                adjusted_loc_parts.append(loc)
                        else:
                            # If the position is out of range, use the original position
                            adjusted_loc_parts.append(loc)
                
                if adjusted_loc_parts:
                    adjusted_locations.append([';'.join(adjusted_loc_parts)])
            
            return adjusted_locations if adjusted_locations else row['location']
        
        # Apply position adjustment logic
        for i, row in train_offset_corrected.iterrows():
            train_offset_corrected.at[i, 'location'] = adjust_location(row)
        
        self.train_offset_corrected = train_offset_corrected
        print("Offset correction completed")
        return self.train_offset_corrected
    
    def process_spaces(self, predictions=None):
        """
        Process spaces (optimized from the second notebook)
        
        The purpose of this function is to clean up spaces in prediction labels through post-processing steps:
        - Remove unnecessary leading and trailing spaces
        - Remove intermediate spaces that are located before and after invalid characters (i.e., no valid characters on either side)
        - Preserve spaces that have valid characters on both sides
        
        Can be used in the preprocessing stage or for post-processing predictions
        """
        print("Processing spaces in text data...")
        
        if not hasattr(self, 'train_offset_corrected'):
            if hasattr(self, 'train_standardized'):
                data = self.train_standardized
            else:
                data = self.train
        else:
            data = self.train_offset_corrected
            
        # Create a working copy
        processed_data = data.copy()
        
        def post_process_spaces(pred, text):
            """
            Process prediction arrays to handle spaces correctly.
            
            Args:
                pred: Prediction array (binary or probability values)
                text: Corresponding text
            
            Returns:
                Processed prediction array
            """
            spaces = ' \n\r\t'
            
            # Ensure matching lengths
            text = text[:len(pred)]
            pred = pred[:len(text)]
            
            # Process boundary spaces
            if text[0] in spaces:
                pred[0] = 0
            if text[-1] in spaces:
                pred[-1] = 0

            # Process internal spaces
            for i in range(1, len(text) - 1):
                if text[i] in spaces:
                    if pred[i] and not pred[i - 1]:  # Space after an invalid character
                        pred[i] = 0

                    if pred[i] and not pred[i + 1]:  # Space before an invalid character
                        pred[i] = 0

                    if pred[i - 1] and pred[i + 1]:  # Space with valid characters on both sides
                        pred[i] = 1
            
            return pred
        
        # If predictions are provided, process them directly
        if predictions is not None:
            processed_predictions = []
            for i, pred in enumerate(predictions):
                if i < len(processed_data):
                    text = processed_data.iloc[i]['pn_history']
                    processed_pred = post_process_spaces(pred, text)
                    processed_predictions.append(processed_pred)
                else:
                    processed_predictions.append(pred)
            return processed_predictions
        
        # Otherwise, process annotation positions in existing data
        # Note: This is a simplified implementation; in reality, more complex logic is needed to handle position information
        # because we need to convert positions to binary arrays, apply space processing, and then convert back to positions
        
        processed_locations = []
        for i, row in processed_data.iterrows():
            text = row['pn_history']
            if pd.isna(text) or text == '':
                processed_locations.append(row['location'])
                continue
                
            # Create binary prediction array
            binary_array = np.zeros(len(text))
            
            # Convert position information to binary array
            for loc_list in row['location']:
                # Fix error: handle different types of loc_list
                if isinstance(loc_list, str):
                    # If it's a string, split by semicolon
                    locations = loc_list.split(';')
                elif isinstance(loc_list, list):
                    # If it's already a list, use directly
                    locations = loc_list
                else:
                    # Skip unknown format
                    continue
                
                for loc in locations:
                    if not isinstance(loc, str):
                        continue
                        
                    # Check if it contains semicolons; if so, further splitting is needed
                    if ';' in loc:
                        sub_locs = loc.split(';')
                        for sub_loc in sub_locs:
                            if ' ' in sub_loc:
                                try:
                                    start, end = map(int, sub_loc.split())
                                    if start < len(binary_array) and end <= len(binary_array):
                                        binary_array[start:end] = 1
                                except ValueError:
                                    print(f"Warning: Cannot parse location string: {sub_loc}")
                    elif ' ' in loc:
                        try:
                            start, end = map(int, loc.split())
                            if start < len(binary_array) and end <= len(binary_array):
                                binary_array[start:end] = 1
                        except ValueError:
                            print(f"Warning: Cannot parse location string: {loc}")
            
            # Apply space processing
            processed_binary = post_process_spaces(binary_array, text)
            
            # Convert processed binary array back to position information
            new_locations = []
            in_span = False
            span_start = -1
            
            for i, val in enumerate(processed_binary):
                if val == 1 and not in_span:
                    # Start new span
                    in_span = True
                    span_start = i
                elif val == 0 and in_span:
                    # End current span
                    new_locations.append(f"{span_start} {i}")
                    in_span = False
            
            # Don't forget to process the ending span
            if in_span:
                new_locations.append(f"{span_start} {len(processed_binary)}")
            
            # Update position information
            if new_locations:
                processed_locations.append([[';'.join(new_locations)]])
            else:
                processed_locations.append([])
        
        # Update processed data
        for i, locs in enumerate(processed_locations):
            if i < len(processed_data):
                processed_data.at[i, 'location'] = locs
        
        # Update class attributes
        if hasattr(self, 'train_offset_corrected'):
            self.train_offset_corrected = processed_data
        elif hasattr(self, 'train_standardized'):
            self.train_standardized = processed_data
        else:
            self.train = processed_data
            
        print("Space processing completed")
        return processed_data
    

    
    def create_folds(self, n_folds=5):
        """
        Create cross-validation folds using GroupKFold (adopted from the second notebook)
        """
        print(f"Creating {n_folds} folds using GroupKFold...")
        self.n_folds = n_folds
        
        if not hasattr(self, 'train_offset_corrected'):
            if hasattr(self, 'train_standardized'):
                data = self.train_standardized
            else:
                data = self.train
        else:
            data = self.train_offset_corrected
        
        # Use GroupKFold based on pn_num groups
        fold = GroupKFold(n_splits=n_folds)
        groups = data['pn_num'].values
        
        for n, (train_index, val_index) in enumerate(fold.split(data, data['location'], groups)):
            data.loc[val_index, 'fold'] = int(n)
            
        data['fold'] = data['fold'].astype(int)
        
        # Update processed data
        if hasattr(self, 'train_offset_corrected'):
            self.train_offset_corrected = data
        elif hasattr(self, 'train_standardized'):
            self.train_standardized = data
        else:
            self.train = data
            
        print(f"Created {n_folds} folds")
        return data
    
    def create_labels_for_scoring(self, df=None):
        """
        Create label format for scoring (adopted from the second notebook)
        """
        if df is None:
            if hasattr(self, 'train_offset_corrected'):
                df = self.train_offset_corrected
            elif hasattr(self, 'train_standardized'):
                df = self.train_standardized
            else:
                df = self.train
        
        # First create a standard format position list
        df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
        
        for i in range(len(df)):
            lst = df.loc[i, 'location']
            if lst:
                # Process different formats of position data
                if isinstance(lst[0], list):
                    # If it's already in list of lists format
                    locations = []
                    for loc_list in lst:
                        for loc in loc_list:
                            locations.append(loc)
                    new_lst = ';'.join(locations)
                elif isinstance(lst[0], str):
                    # If it's in string list format
                    new_lst = ';'.join(lst)
                else:
                    # Unknown format, skip
                    continue
                    
                df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[[\"{new_lst}\"]]')
        
        # Create labels
        truths = []
        for location_list in df['location_for_create_labels'].values:
            truth = []
            if len(location_list) > 0:
                location = location_list[0]
                for loc in [s.split() for s in location.split(';')]:
                    if len(loc) >= 2:  # Ensure there are start and end positions
                        start, end = int(loc[0]), int(loc[1])
                        truth.append([start, end])
            truths.append(truth)
            
        return truths
        
    def pred_to_chars(self, token_type_logits, len_token, max_token, offset_mapping, text, feature_num):
        """
        Convert model token-level predictions to character-level predictions
        
        This function handles special medical notation such as "yof" (years old female) and "yom" (years old male)
        
        Args:
            token_type_logits: Model's token-level predictions (logits)
            len_token: Actual length of the token sequence
            max_token: Maximum length of the token sequence
            offset_mapping: Mapping from tokens to original text characters
            text: Original text
            feature_num: Feature number being processed
            
        Returns:
            tuple: (character-level predictions, original text)
        """
        # Truncate to actual token length
        token_type_logits = token_type_logits[:len_token]
        offset_mapping = offset_mapping[:len_token]
        
        # Initialize character-level predictions
        char_preds = np.ones(len(text)) * -1e10
        
        # Iterate through each token mapping
        for i, (start, end) in enumerate(offset_mapping):
            # Special handling for "yof" (age + female)
            if text[start:end] == 'of' and start > 0 and text[start-1:end] == 'yof':
                if feature_num in self.feature_female:
                    # If the feature is female-related, mark the last character
                    char_preds[end-1:end] = 1
                elif feature_num in self.feature_year:
                    # If the feature is age-related, use the prediction from the previous token
                    char_preds[start:start+1] = token_type_logits[i-1]
                else:
                    # For other cases, use the current token's prediction
                    char_preds[start:end] = token_type_logits[i]
            
            # Special handling for "yom" (age + male)
            elif text[start:end] == 'om' and start > 0 and text[start-1:end] == 'yom':
                if feature_num in self.feature_male:
                    # If the feature is male-related, mark the last character
                    char_preds[end-1:end] = 1
                elif feature_num in self.feature_year:
                    # If the feature is age-related, use the prediction from the previous token
                    char_preds[start:start+1] = token_type_logits[i-1]
                else:
                    # For other cases, use the current token's prediction
                    char_preds[start:end] = token_type_logits[i]
            
            # Standard processing for other tokens
            else:
                char_preds[start:end] = token_type_logits[i]
                
        return (char_preds, text)
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create train/test split (if test set is not provided)
        """
        from sklearn.model_selection import train_test_split
        
        if self.test is not None:
            print("Test data already provided, skipping split.")
            return
            
        print(f"Creating train/test split with test_size={test_size}...")
        
        if hasattr(self, 'train_offset_corrected'):
            data = self.train_offset_corrected
        elif hasattr(self, 'train_standardized'):
            data = self.train_standardized
        else:
            data = self.train
        
        # Use stratified sampling, maintaining the proportion of each pn_num
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['pn_num']
        )
        
        self.train_final = train_data.reset_index(drop=True)
        self.test = test_data.reset_index(drop=True)
        
        print(f"Split completed. Train: {len(self.train_final)} rows, Test: {len(self.test)} rows")
        return self.train_final, self.test
    
    def save_processed_data(self):
        """
        Save processed data
        """
        print("Saving processed data...")
        
        if hasattr(self, 'train_offset_corrected'):
            final_train = self.train_offset_corrected
        elif hasattr(self, 'train_standardized'):
            final_train = self.train_standardized
        else:
            final_train = self.train
        
        if not hasattr(self, 'train_final'):
            self.train_final = final_train
        
        # Save training data
        train_output_path = os.path.join(self.output_dir, "train_processed.csv")
        self.train_final.to_csv(train_output_path, index=False)
        print(f"Saved processed train data to {train_output_path}")
        
        # Save test data (if available)
        if self.test is not None:
            test_output_path = os.path.join(self.output_dir, "test_processed.csv")
            self.test.to_csv(test_output_path, index=False)
            print(f"Saved processed test data to {test_output_path}")
        
        # Save label information (for scoring)
        truths = self.create_labels_for_scoring(self.train_final)
        labels_output_path = os.path.join(self.output_dir, "train_labels.npy")
        # Use dtype=object to save arrays of irregular shape
        np.save(labels_output_path, np.array(truths, dtype=object))
        print(f"Saved labels to {labels_output_path}")
        
        # Save fold information
        folds_output_path = os.path.join(self.output_dir, "folds.npy")
        np.save(folds_output_path, self.train_final['fold'].values)
        print(f"Saved fold information to {folds_output_path}")
        
        return True
    
    def run_full_pipeline(self):
        """
        Run the complete data processing pipeline
        """
        print("Starting full data processing pipeline...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Preprocess features
        self.preprocess_features()
        
        # 3. Parse annotations
        self.parse_annotations()
        
        # 4. Merge data
        self.merge_data()
        
        # 5. Check annotation integrity (instead of manual corrections)
        self.check_annotation_integrity()
        
        # 6. Standardize medical text
        self.standardize_medical_text()
        
        # 7. Correct offsets
        self.correct_offsets()
        
        # 8. Process spaces
        self.process_spaces()
        
        # 9. Create folds
        self.create_folds(n_folds=self.n_folds)
        
        # 11. Create train/test split (if needed)
        if self.test is None:
            self.create_train_test_split()
        
        # 12. Save processed data
        self.save_processed_data()
        
        print("Full data processing pipeline completed!")
        return self.train_final, self.test

# Usage example
if __name__ == "__main__":
    processor = NBMEDataProcessor(
        data_dir=r"C:\Users\SIMON\Desktop\NLP\nbme-score-clinical-patient-notes", 
        output_dir=r"C:\Users\SIMON\Desktop\NLP\processed"
    )
    
    # Run the complete processing pipeline
    train_data, test_data = processor.run_full_pipeline()
    
    print(f"Processed train data shape: {train_data.shape}")
    if test_data is not None:
        print(f"Processed test data shape: {test_data.shape}")