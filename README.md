# EasyFaiss
Wrapper for few typical use cases

# Classification
Great for semantically distinct classes.
Terrible otherwise.

Classifier has been tested on BANKING77 - Dataset composed of online banking queries annotated with their corresponding intents.

| Example Query                  | Intent           |
|--------------------------------|------------------|
| I am still waiting on my card? | card_arrival     |
| I think my card is broken      | card_not_working |

Running example script banking77.py should return precision, recall and f1-score for each label and following micro and weighted averages

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| macro avg    | 0.92      | 0.88   | 0.90     | 3072    |
| weighted avg | 0.93      | 0.89   | 0.91     | 3072    |

Label Errors in BANKING77 - Cecilia Ying, Stephen Thomas (https://aclanthology.org/2022.insights-1.19/) highlights a problem with dataset, but it's still a good showcase for this specific use case.
