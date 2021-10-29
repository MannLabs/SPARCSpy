from datetime import datetime
import os
import warnings
import shutil

class Logable(object):
    """object which can create log entries.
        
    Args:
        directory (str): A directory must be set in every descendent before log can be called.
        
        DEFAULT_LOG_NAME (str, optional, default ``processing.log``): Default log file.
        
        DEFAULT_FORMAT (str): Date and time format used for logging.
    """
    
    DEFAULT_LOG_NAME = "processing.log"
    DEFAULT_FORMAT = "%d/%m/%Y %H:%M:%S"
    
    def __init__(self, debug=False):
        
        
        self.debug = debug
        
    def log(self, message):
        """log a message

        Args:
            message (str, list(str), dict(str)): Strings are s
        """
        
        if not hasattr(self, 'directory'):
            raise ValueError("Please define a valid self.directory in every descendend of the Logable class")

        if isinstance(message, str):
            lines = message.split("\n")
            
        if isinstance(message, list):
            lines = message
            
        if isinstance(message, dict):   
            lines = []
            for key, value in message.items():
                lines.append(f"{key}: {value}") 
                
        else:
            try:
                lines = [str(message)]
            except:
                self.log("unknown type during loging")
                return
        
        for line in lines:
            log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)
            with open(log_path, "a") as myfile:
                myfile.write(self.get_timestamp() + line +" \n")

            if self.debug:
                print(self.get_timestamp() + line)
                    
    def get_timestamp(self):
        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime(self.DEFAULT_FORMAT)  
        return "[" + dt_string + "] "
    
class ProcessingStep(Logable):
    
    """Processing step living in its own subdirectory of the project.
        
    Args:
        config (dict): Config file which is passed by the Project class when called. Is loaded from the project based on the name of the class.
        
        directory (str): Directory which should be used by the processing step. A subdirectory of the project directory is passed by the project class when called. The directory will be newly created if it does not exist yet.
        
        intermediate_output (bool, optional, default ``False``): When set to True intermediate outputs will be saved where applicable.
            
        debug (bool, optional, default ``False``): When set to True debug outputs will be printed where applicable. 
            
        overwrite (bool, optional, default ``True``): When set to True, the processing step directory will be delted and newly created when called.

    """
    
    def __init__(self,
                 config, 
                 directory, 
                 debug=False, 
                 intermediate_output=False,
                 overwrite=True):
        
        super().__init__()
        
        
        self.debug = debug
        self.overwrite = overwrite
        self.intermediate_output = intermediate_output
        self.directory = directory
        self.config = config
        
        
    def __call__(self, *args,
                 debug=None, 
                 intermediate_output=None, 
                 overwrite=None, **kwargs):
    
        """object which can create log entries.
        
        Args:
            
            intermediate_output (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True intermediate outputs will be saved where applicable.
            
            debug (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True debug outputs will be printed where applicable. 
                
            overwrite (bool, optional, default ``None``): Allows overriding the value set on initiation. When set to True, the processing step directory will be delted and newly created when called.

        """
    
        # set flags if provided
        self.debug = debug if debug is not None else self.debug
        self.overwrite = overwrite if overwrite is not None else self.overwrite
        self.intermediate_output = intermediate_output if intermediate_output is not None else self.intermediate_output

        # remove directory for processing step if overwrite is enabled
        if self.overwrite:
            if os.path.isdir(self.directory):
                shutil.rmtree(self.directory)

        # create directory for processing step 
        if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

        process = getattr(self, "process", None)
        if callable(process):
            self.process(*args,**kwargs)
        else:
            warnings.warn("no process method defined")
            