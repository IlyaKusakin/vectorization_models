# -*- coding: utf-8 -*-
"""
Module with fixtures for unit-tests.

@author: Kusakin Ilya
"""
import pytest
from source.corpora import Corpora

corpora_file = 'data\\corpora.csv'

@pytest.fixture(scope='module')
def corpora_load():
    return Corpora(corpora_file)    


