def is_in_pretrained(pretrained_l, level):
    for p in pretrained_l:
        if level == p['level']:
            return True
    return False

def get_pretrained_for(pretrained_l, level):
    for p in pretrained_l:
        if level == p['level']:
            return p
    
    return None