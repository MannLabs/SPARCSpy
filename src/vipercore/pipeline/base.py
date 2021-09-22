from datetime import datetime

class Logable(object):
    
    def __init__(self, debug=False):
        
        self.log_path = None
        self.debug = debug
        
    def log(self, message):
        
        if self.log_path is None:
            raise ValueError("Please define a valid self.log_path in every descendend of the Logable class")

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
                with open(self.log_path, "a") as myfile:
                    myfile.write(self.get_timestamp() + line +" \n")
                
                if self.debug:
                    print(self.get_timestamp() + line)
                    
    def get_timestamp(self):
        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  
        return "[" + dt_string + "] "