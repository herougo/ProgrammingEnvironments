# Example PR comments
- basic
  - magic variables
  - unecessary computation (ie unzip dict and back)
  - changing a function and forgetting to update other places that use it
  - hacky solution (the tests pass, but it does not address the real feature)
- hard to read
  - missing comments
  - overly pythonic
  - PR usually cap length at 50-200 (otherwise split into separate PRs)
  - long variable names instead of a comment
  - when reading a test case, you should be able to understand it wuthout going to helper functions
    - possibly bad: assertEqual in helper function
  - pretty print json in snapshot for easier readability
- code style
  - not use snapshot for small arrays (easier review and prevents the snapshot files getting clogged)
- specific
  - modifying protos is wrong
- should fail early

# THE DEBUGGING GUIDE (Hopefully)
- finding logs
  - search session id
    - find in context
      - look for e.g. POST response
      - ctr + f using etc price
- is it an issue?
- determine the service first
- debug locally (10m)

# After coding
- look over for readability, bugs, unecessary code, etc

# Sprint Lessons
- large PRs built up without getting merged by the end of the sprint
- break up migrations (or other complicated logic) into multiple PRs
  - e.g. 1st PR add code from other repo, 2nd PR request logic, 3rd PR response logic
  - makes review easier (more lines = more things you're comparing)
- for complex issues spanning multiple packages and many edge cases, I should've created a sheet with all the different edge cases as rows and the pip packages as columns
- usually prioritize new production bugs over current work
- always look into the cause of the issue (don't just look at the code and assume you know to save time)
- when modifying redis (accidentally deleted in prod)
  - keep track of original value
  - make sure it's not used in prod
  - assume the worst: used in prod, not back up available
- not testing changes early enough
  - need to get tickets done
- should probably be testing big changes locally first before merging
- side projects all weekend can lead to fatigue Monday morning

** PyCharm

# PyCharm Need to Be Able to Do
load repo from command line
- Cmd + V can fail for Mac
  - Use mouse paste
keyboard shortcut code snippets
files side by side

# PyCharm Keyboard Shortcut Changes
cmd + shift + t	reopen closed tab
cmd + o: open

Preferences > Editor > General > Code Folding > Collapse by Default: Uncheck Imports

# Mac Settings Preferences
System Preferences > Mission Control > Uncheck: Automatically rearrange Spaces based on most recent use
System Preferences > Spotlight > Uncheck all but Documents, Folders, Images, PDF Documents, Spreadsheets, System Preferences


### Troubleshooting

# Troubleshooting Mocking URL Fail
- responses
  - did you response.activate the containing function?
  - Are you trying to responses.add in multiple functions at a time?
- mock
  - start()
  - target the source of its use?
  - Is it called again by something else you don't care about?

# Troubleshooting Pdb fail
- are you using an alias which passes text to pdb?

# Troubleshooting Tests Fail
- Did you install dependencies?
- Did you get the environment variables?
- Are you  running the necessary DB, cache, etc?


# Things to do when early:
- code improve
  - toreuse_script (and commands)
  - implement htools desired functions
  - (learn stuff)
- learn backend
  - read early pull requests
  - move forward from to learn sheet
  - learn code design/technology application
    - how do they organize code for ___ (e.g. patch scripts, deployment, data formats, etc)?
    - how can you use a technology (e.g. redis)?
    - look at helper functions (e.g. encryption, parse protobuf before passing to function)

# To Learn
- Stress/load testing
- Setting up scalable systems (e.g. AWS, Docker, etc)

Generic Learning goals
- learn backend
- improve debugging/coding (or code base learning) speed

Specific goals
- be able to design an entire service repo from scratch
- learn automation details (ie automatic snapshot code commits via Gitlab, git test)
- deploying process


