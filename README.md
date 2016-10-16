# pandastransformers

Some sklearn api conforming transformer objects that (typically) return pandas dataframes.
An example application to the kaggle competition *House Prices - Advanced Regression Techniques* will be provided in the future.

Current issues:
* DataFrameFeatureUnion is not working with GridSearchCV. This seems to be due to missing get_params functions. Compare to *Raschka - Python Machine Learning* and his remarks converning the MajorityVoteClassifier class.
* Rework the Notebook and construct a blog entry
