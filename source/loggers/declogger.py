# -*- coding: utf-8 -*-
"""
Module with custom decorators for logging

@author: PS42
"""

import logging
import logging.config

class FunctionLogger():
    """
    
    Custom logger class that calls like function decorator.
    
    Uses retuned from logging.getLogger(...) logger. 
   
    """
    def __init__(self, name=__name__, 
                 config_file=r'source\loggers\logger_configs.conf'): 
        
        self.config_file = config_file
        logging.config.fileConfig(fname=self.config_file, disable_existing_loggers=False)
        self.logger = logging.getLogger(name)
        
    def __call__(self, func):
        """Equal to decorator function."""
        def logging(*args, **kargs):     
            log_msg = func.__name__ + ' function was called.'
            try:
                func(*args, **kargs)
                self.logger.info(log_msg)
                
            except Exception as e:
                msg = 'in function ' + func.__name__
                self.logger.error(e)
                self.logger.error(msg)
        return logging
         
        