import re
class inspection:
    def __init__(self):
        pass
    
    def validate(self,input):
        out = input
        if not self.is_only_numbers(input['national_code']):
            out['national_code'] = 'Unknown'
        if not self.is_date_format(input['expire_date']):
            out['expire_date'] = 'Unknown'
        if not self.is_date_format(input['birth_date']):
            out['birth_date'] = 'Unknown'
        if not self.is_only_letters(input['first_name']):
            out['first_name'] = 'Unknown'
        if not self.is_only_letters(input['last_name']):
            out['last_name'] = 'Unknown'
        if not self.is_only_letters(input['father_name']):
            out['father_name'] = 'Unknown'
        return out
    
    def is_date_format(self,text):
        if text=='Unknown':
            return False
        
        pattern = r'^(\d{4})/(\d{2})/(\d{2})$'
        match = re.match(pattern, text)
        
        if not match:
            return False
        
        year, month, day = map(int, match.groups())
        
        if year < 1300 or year > 1450:  
            return False
        
        if month < 1 or month > 12:
            return False
        
        if day < 1 or day > 31:
            return False
        

        if month <= 6 and day > 31:
            return False
        elif 7 <= month <= 11 and day > 30:
            return False
        elif month == 12 and day > 29:  
            return False
        
        return True
    
    def is_only_numbers(self,text):
        if text=='Unknown':
            return False
        return bool(re.match(r'^\d+$', text))
    
    def is_only_letters(self,text):
        if text=='Unknown':
            return False
        return not bool(re.search(r'[\d/]', text))
    