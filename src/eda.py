import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from collections import Counter

# Try multiple possible paths
paths_to_try = [
    r'C:\Users\SIMON\Desktop\NLP\nbme-score-clinical-patient-notes',  # Absolute path
    os.path.join(os.getcwd(), 'nbme-score-clinical-patient-notes'),   # Subdirectory in current working directory
    './nbme-score-clinical-patient-notes',                            # Relative path
    '../nbme-score-clinical-patient-notes',                           # Subdirectory in parent directory
]

# Set path (will be verified in the load_data function)
BASE_DIR = paths_to_try[0]  # Default to first path

def load_data():
    """Load all data files"""
    global BASE_DIR
    
    # Traverse all possible paths
    for path in paths_to_try:
        try:
            print(f"Trying to read from path: {path}")
            if not os.path.exists(path):
                print(f"  Path does not exist, skipping")
                continue
                
            # Try to read files
            patient_notes_path = os.path.join(path, 'patient_notes.csv')
            train_path = os.path.join(path, 'train.csv')
            features_path = os.path.join(path, 'features.csv')
            
            print(f"  Trying to read: {patient_notes_path}")
            patient_notes = pd.read_csv(patient_notes_path)
            
            print(f"  Trying to read: {train_path}")
            train = pd.read_csv(train_path)
            
            print(f"  Trying to read: {features_path}")
            features = pd.read_csv(features_path)
            
            # If all files are successfully read, set BASE_DIR and return data
            print(f"\nSuccessfully read data from path: {path}")
            BASE_DIR = path
            
            print(f"Patient notes shape: {patient_notes.shape}")
            print(f"Train shape: {train.shape}")
            print(f"Features shape: {features.shape}")
            
            return patient_notes, train, features
            
        except FileNotFoundError as e:
            print(f"  File not found: {e}")
            continue
        except Exception as e:
            print(f"  Error when trying path {path}: {e}")
            continue
    
    # If all path attempts fail
    print("\nAll path attempts failed.")
    print("Current working directory:", os.getcwd())
    print("Current directory contents:")
    for item in os.listdir():
        print(f" - {item}")
        
    # Check if there are directories starting with nbme
    nbme_dirs = [d for d in os.listdir() if os.path.isdir(d) and 'nbme' in d.lower()]
    if nbme_dirs:
        print("\nDiscovered possible NBME data directories:")
        for d in nbme_dirs:
            print(f" - {d}")
    
    raise FileNotFoundError("Cannot find data files, please check the path and try again")

def analyze_patient_history_word_count(patient_notes):
    """Analyze word count distribution in patient history"""
    print("Column names of patient notes data:", patient_notes.columns.tolist())
    
    # Find columns that may contain patient history
    history_columns = [col for col in patient_notes.columns if 'history' in col.lower() or 'note' in col.lower() or 'text' in col.lower()]
    
    if 'pn_history' in patient_notes.columns:
        history_column = 'pn_history'
    elif history_columns:
        history_column = history_columns[0]
        print(f"Using {history_column} column as patient history")
    else:
        # Assume the longest text column is the history column
        text_lengths = {col: patient_notes[col].astype(str).apply(len).mean() for col in patient_notes.columns}
        history_column = max(text_lengths, key=text_lengths.get)
        print(f"Based on text length, guessing {history_column} is the patient history column")
    
    # Print some sample data
    print("\nFirst few rows of patient notes data:")
    print(patient_notes.head())
    print(f"\nExample of {history_column} column:")
    print(patient_notes[history_column].iloc[0])
    
    # Calculate word count for each history
    patient_notes['word_count'] = patient_notes[history_column].astype(str).apply(lambda x: len(x.split()))
    
    # Print word count range
    print(f"\nWord count range: {patient_notes['word_count'].min()} to {patient_notes['word_count'].max()}")
    
    # Create histogram of word count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(patient_notes['word_count'], kde=True)
    plt.title('Patient History Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('patient_history_word_count.png')
    
    # Print some statistics
    print("\nPatient History Word Count Statistics:")
    print(patient_notes['word_count'].describe())
    
    return patient_notes

def analyze_train_annotations(train, features):
    """Analyze annotation distribution in training data and feature word count distribution"""
    print("Column names of training data:", train.columns.tolist())
    print("Column names of feature data:", features.columns.tolist())
    
    # Check and determine the feature text column
    if 'feature_text' in features.columns:
        feature_text_col = 'feature_text'
    else:
        # Assume feature text is in second column
        feature_text_col = features.columns[1] if len(features.columns) > 1 else features.columns[0]
        print(f"Using {feature_text_col} as feature text column")
    
    # Print some sample data
    print("\nFirst 5 rows of training data:")
    print(train.head())
    print("\nFirst 5 rows of feature data:")
    print(features.head())
    
    # Analyze feature text
    try:
        # Note that feature text may be connected using hyphens, such as "Family-history-of-MI"
        # Calculate the number of terms separated by hyphens
        features['term_count'] = features[feature_text_col].astype(str).apply(lambda x: len(x.split('-')))
        
        # Output some feature text examples and their term counts
        print("\nFeature text examples and their term counts:")
        for i in range(min(10, len(features))):
            text = features.iloc[i][feature_text_col]
            count = features.iloc[i]['term_count']
            print(f"{text}: {count} terms")
        
        # Check if there are features with only 1 term
        features_with_one_term = features[features['term_count'] == 1]
        if len(features_with_one_term) > 0:
            print(f"\nFound {len(features_with_one_term)} features with only 1 term:")
            print(features_with_one_term[feature_text_col].values)
        else:
            print("\nNo features with only 1 term in the data")
        
        # Calculate the number of features for each term count
        term_count_distribution = features['term_count'].value_counts().sort_index().reset_index()
        term_count_distribution.columns = ['term_count', 'feature_count']
        
        # Print original distribution
        print("\nOriginal term count distribution:")
        print(term_count_distribution)
        
        # Check if rows with term count of 1 are missing
        if 1 not in term_count_distribution['term_count'].values:
            print("\nWarning: Features with term count of 1 are missing in statistics, attempting to fix...")
            # Manually calculate the number of features with term count of 1
            count_one_term = len(features_with_one_term)
            one_term_row = pd.DataFrame({'term_count': [1], 'feature_count': [count_one_term]})
            term_count_distribution = pd.concat([one_term_row, term_count_distribution]).sort_values('term_count').reset_index(drop=True)
            print("Distribution after fix:")
            print(term_count_distribution)
        
        # Ensure all possible term counts are included (from 1 to max)
        max_term_count = term_count_distribution['term_count'].max()
        all_term_counts = set(range(1, max_term_count + 1))
        existing_term_counts = set(term_count_distribution['term_count'])
        missing_term_counts = all_term_counts - existing_term_counts
        
        if missing_term_counts:
            print(f"\nFound missing term counts: {missing_term_counts}")
            # Add missing term counts with feature count of 0
            missing_rows = pd.DataFrame({
                'term_count': list(missing_term_counts),
                'feature_count': [0] * len(missing_term_counts)
            })
            term_count_distribution = pd.concat([term_count_distribution, missing_rows]).sort_values('term_count').reset_index(drop=True)
        
        # Print final distribution
        print("\nFinal term count distribution:")
        print(term_count_distribution)
        
        # Use matplotlib to directly draw a bar chart, rather than through seaborn
        plt.figure(figsize=(14, 7))
        
        # Get term counts and corresponding feature counts
        term_counts = term_count_distribution['term_count'].values
        feature_counts = term_count_distribution['feature_count'].values
        
        # Create bar chart
        bars = plt.bar(term_counts, feature_counts, color='steelblue', width=0.7)
        
        # Add value labels to bar chart
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    '%d' % int(height), ha='center', va='bottom')
        
        # Set chart title and axis labels
        plt.title('Number of Features vs. Number of Terms', fontsize=14)
        plt.xlabel('Number of Terms in Feature (Hyphen-Separated)', fontsize=12)
        plt.ylabel('Number of Features', fontsize=12)
        
        # Set x-axis ticks to ensure all term counts are displayed
        plt.xticks(term_counts)
        
        # Add grid lines
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adjust x-axis range to ensure all bars are visible
        plt.xlim(0.5, max(term_counts) + 0.5)
        
        plt.tight_layout()
        plt.savefig('feature_term_count_distribution.png')
        
        # Print confirmation message
        print("\nFeature term count statistics:")
        print(f"Term count distribution:")
        print(term_count_distribution)
        
        # Print statistical information
        print("\nFeature term count statistics:")
        print(f"Term count distribution:\n{term_count_distribution}")
        print("\nDescriptive statistics of term count:")
        print(features['term_count'].describe())
        
        # If there is an annotation column in the training data, analyze the relationship between annotation count and frequency
        if 'annotation' in train.columns:
            # Count the number of occurrences for each annotation
            annotation_counts = train['annotation'].value_counts().reset_index()
            annotation_counts.columns = ['annotation', 'occurrence_count']
            
            # Calculate how many annotations have each count value
            count_distribution = annotation_counts['occurrence_count'].value_counts().sort_index().reset_index()
            count_distribution.columns = ['occurrence_count', 'num_annotations']
            
            # Draw a chart showing the relationship between annotation count and frequency
            plt.figure(figsize=(12, 6))
            sns.barplot(x='occurrence_count', y='num_annotations', data=count_distribution)
            plt.title('Number of Annotations vs. Annotation Count')
            plt.xlabel('Annotation Count (How Many Times an Annotation Appears)')
            plt.ylabel('Number of Annotations')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('annotation_count_vs_frequency.png')
            
            # Print annotation count statistics
            print("\nAnnotation count statistics:")
            print(f"Average occurrences per annotation: {annotation_counts['occurrence_count'].mean():.2f}")
            print(f"Maximum occurrences of an annotation: {annotation_counts['occurrence_count'].max()}")
            print(f"Annotation count distribution:\n{count_distribution}")
            
            return {
                'term_count_distribution': term_count_distribution,
                'annotation_count_distribution': count_distribution
            }
        else:
            print("No annotation column in training data, cannot analyze annotation counts")
            return {
                'term_count_distribution': term_count_distribution
            }
        
    except Exception as e:
        print(f"Error analyzing feature data: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_annotation_counts(train):
    """Analyze the comparison between annotation count and annotations per row"""
    print("Column names of training data:", train.columns.tolist())
    
    # Determine grouping column based on data structure
    if 'case_num' in train.columns and 'pn_num' in train.columns:
        print("Using combination of 'case_num' and 'pn_num' as grouping key")
        # Create combination key
        train['case_pn_key'] = train['case_num'].astype(str) + "_" + train['pn_num'].astype(str)
        group_key = 'case_pn_key'
    elif 'case_id' in train.columns:
        print("Using 'case_id' column as grouping key")
        group_key = 'case_id'
    elif 'id' in train.columns:
        print("Using 'id' column as grouping key")
        group_key = 'id'
    else:
        # Find possible alternative columns
        possible_id_columns = [col for col in train.columns if 'id' in col.lower() or 'case' in col.lower()]
        if possible_id_columns:
            print(f"Using '{possible_id_columns[0]}' column as grouping key")
            group_key = possible_id_columns[0]
        else:
            print("No ID column found, using index as alternative")
            train['temp_case_id'] = train.index
            group_key = 'temp_case_id'
    
    # Print some sample data rows to help understand the data structure
    print("\nFirst 5 rows of training data:")
    print(train.head())
    
    # Calculate number of annotations in each case/patient note combination
    annotation_counts = train.groupby(group_key).size().reset_index(name='annotation_count')
    
    # Print some data after grouping
    print("\nAnnotation counts calculated by grouping key (first 5 rows):")
    print(annotation_counts.head())
    
    # Calculate the frequency of each annotation count (i.e., how many combinations contain 1 annotation, how many combinations contain 2 annotations, etc.)
    count_distribution = annotation_counts['annotation_count'].value_counts().sort_index()
    count_distribution_df = count_distribution.reset_index()
    count_distribution_df.columns = ['annotations_per_combination', 'count']
    
    # Print distribution
    print("\nAnnotation count distribution:")
    print(count_distribution)
    
    # Draw a comparison chart of "annotation count" vs. "number of combinations with that count"
    plt.figure(figsize=(10, 6))
    sns.barplot(x='annotations_per_combination', y='count', data=count_distribution_df)
    plt.title('Number of Case-Patient Note Combinations vs. Number of Annotations')
    plt.xlabel('Number of Annotations per Case-Patient Note Combination')
    plt.ylabel('Number of Combinations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('annotation_count_distribution.png')
    
    # Print some statistics
    print("\nAnnotation Count Statistics:")
    print(f"Average annotations per combination: {annotation_counts['annotation_count'].mean():.2f}")
    print(f"Median annotations per combination: {annotation_counts['annotation_count'].median()}")
    print(f"Max annotations per combination: {annotation_counts['annotation_count'].max()}")
    print(f"Total number of combinations: {len(annotation_counts)}")
    print(f"Distribution of annotations per combination:\n{count_distribution}")
    
    return annotation_counts, count_distribution

def main():
    """Main function"""
    print("Loading data...")
    patient_notes, train, features = load_data()
    
    print("\nAnalyzing patient history word count...")
    patient_notes = analyze_patient_history_word_count(patient_notes)
    
    print("\nAnalyzing train annotations...")
    feature_data = analyze_train_annotations(train, features)
    
    print("\nAnalyzing annotation counts...")
    annotation_counts, count_distribution = analyze_annotation_counts(train)
    
    print("\nAnalysis complete. Visualizations saved as PNG files.")
    
    return {
        'patient_notes': patient_notes,
        'feature_data': feature_data,
        'annotation_counts': annotation_counts,
        'count_distribution': count_distribution
    }

if __name__ == "__main__":
    main()