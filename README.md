# Console app for word vectorization via Word2Vec and TF-IDF
Implemented on numpy calculations and lil_matrixes from scipy library.

#### Result of script running
Json-file with dictionary "word": vector .

#### Run to install all neccessary libs:
```
pip install -r requirements.txt
```

#### Run to test the application code:
```
pytest unit_tests.py
```
![tests_screenshot](https://github.com/IlyaKusakin/vectorization_models/blob/main/images/tests.jpg)

#### Run to see the help-message with instruction for each argument:
```
python main.py --help
```
![help_screenshot](https://github.com/IlyaKusakin/vectorization_models/blob/main/images/help.jpg)

#### Base example of usage:
```
python main.py --model==Word2Vec
```
![example_screenshot](https://github.com/IlyaKusakin/vectorization_models/blob/main/images/example.jpg)


