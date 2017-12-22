kr_path = '/Users/hromel/Documents/GitHub/CodeBankAlpha.wiki/'
krtxt = kr_path + 'cb.txt'

kr_github = 'https://github.com/herougo/CodeBankAlpha/wiki/'

'''
Hi
     1
-
     2
  - 
     3
- 
     2





'''


def fill_folder_names(i, folder_names, page_names, children, in_home, so_far=''):
    use_folder = in_home[i] or len(children[i])
    if use_folder:
        so_far += page_names[i].replace(' ', '-') + '/'
    folder_names[i] = so_far
    for child_i in children[i]:
        fill_folder_names(child_i, folder_names, page_names, children, in_home, so_far)
    

def parse_wiki_structure(path, github_prefix, repo_prefix):
    lines = fileToLines(path)
    n = len(lines)
    
    children = [[] for x in range(n)]
    content = [''] * n
    page_names = [''] * n
    file_names = [''] * n
    in_home = [False] * n
    home_pages = []
    folder_names = [''] * n
    
    file_stack_i = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        elif stripped.startswith('*'):
            content[file_stack_i[-1]] += stripped[1:].strip() + '\n'
        elif stripped.startswith('-'):
            n_front_spaces = len(re.match('^ *', line)[0])
            depth = n_front_spaces // 2 + 2
            if depth == len(file_stack_i):
                # keep kepth the same
                file_stack_i.pop()
            elif depth == len(file_stack_i) + 1:
                # increase depth by 1
                pass
            elif depth < len(file_stack_i):
                # decrease depth
                for j in range(len(file_stack_i) - depth + 1):
                    file_stack_i.pop()
            else:
                raise BaseException(line)
            children[file_stack_i[-1]].append(i)
            file_stack_i.append(i)
            page_names[i] = stripped[1:].strip()
            
        else: # normal page
            page_names[i] = stripped
            file_stack_i = [i]
            in_home[i] = True
            home_pages.append(i)
        
    # calculate file_names
    file_names = list(map(lambda s: s.replace(' ', '-'), page_names))
    
    for i in range(n):
        if len(children[i]) > 0:
            to_add = ['\n## Children Pages\n']
            for child_i in children[i]:
                link_path = github_prefix + file_names[child_i]
                link_text = page_names[child_i]
                to_add.append(f'- [{link_text}]({link_path})')
            content[i] += '\n'.join(to_add)
            
    # check bad file names
    BAD_FILE_CHARS = '<>:"/\\|?*'
    for name in file_names:
        if len(set(name) & set(BAD_FILE_CHARS)) > 0:
            print('Bad File Name:', name)
            
    # fill folder names
    folder_names = [''] * n
    for i in home_pages:
        fill_folder_names(i, folder_names, page_names, children, in_home)
        
    # create missing folders
    for folder_name in folder_names:
        folder_path = os.path.join(repo_prefix, folder_name)
        if not folderExists(folder_path):
            createFolder(folder_path)
    
    to_print = myflatten(list(zip(file_names, page_names, folder_names, content)))
    myprint(to_print)
    #myprint(in_home)
    return children, content, page_names, file_names, in_home, home_pages, folder_names

result = parse_wiki_structure(krtxt, kr_github, kr_path)    


# Create Files
children, content, page_names, file_names, in_home, home_pages, folder_names = result
for i in range(len(content)):
    md_path = os.path.join(kr_path + folder_names[i], file_names[i] + '.md')
    stringToFile(file_name=md_path, text=content[i])

# create sidebar
sidebar_path = os.path.join(kr_path, '_Sidebar.md')
to_add = ['Home\n']
for i in home_pages:
    link_path = kr_github + file_names[i]
    link_text = page_names[i]
    to_add.append(f'- [{link_text}]({link_path})')
print('\n'.join(to_add))

stringToFile(file_name=sidebar_path, text='\n'.join(to_add))