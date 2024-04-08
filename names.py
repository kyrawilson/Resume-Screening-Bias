import pandas as pd

#Lol you forgot to even check if the line was in the subset so you just did all of them again whoops
names = ['Anne Baker', 'Emily Murphy', 'Kristen Sullivan', 
         'Brendan Baker', 'Greg Murphy', 'Matthew Sullivan',
         'Ebony Jones', 'Lakisha Robinson', 'Tanisha Washington',
         'Darnell Jones', 'Kareem Robinson', 'Rasheed Washington', '']

universities = ['Howard University', 'Florida Agricultural and Mechanical University', 
                'North Carolina Agricultural and Technical State University',
                'Delaware State University', 'Morgan State University', 'Bowie State University',
                'Harvard University', 'Stanford University', 'Duke University',
                'University of Massachusetts–Amherst', 'University of California–Riverside', 
                'University of North Carolina–Greensboro', '']

for name in names:
    for univ in universities:
        if name != '' and univ != '':
            new = f'{name}\\n{univ}\\n'
        elif name == '' and univ != '':
            new = f'{univ}\\n'
        elif univ == '' and name != '':
            new = f'{name}\\n'
        else:
            continue
        with open(f'resume_prefixes/{name.replace(" ", "")}_{univ.replace(" ", "")}.txt', 'w') as f2:
            f2.write(new)