CLINICAL_FEATURES = [

    "PSA",

    "age"

]

MOLECULAR_FEATURES = [

    "Expression_ASS1",

    "Expression_CPS1",

    "Methylation_CPS1",

    "Methylation_CPS1",

    "SNP_1", 
    
    "SNP_2", 
  
    "SNP_3"

]


COMBINED_FEATURES = CLINICAL_FEATURES + MOLECULAR_FEATURES

TARGET_COLUMN = "Group"

RANDOM_STATE = 42

OUTER_FOLDS = 10

INNER_FOLDS = 5

BOOTSTRAP_ITERATIONS = 2000
