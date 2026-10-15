# To-dos

Goal: Produce a prompt for an agentic LLM to get the following points done in the current working directory.
First, explore the directory without any goal-driven intent: Get an overview and understanding of the project / context.

1. First task.
2. Second task.

If there is any uncertainty, ask.
It is way better to ask redundant questions vs running with potentially misunderstood requests.

After producing the first prompt, ask the user if it suits their intent.
If they confirm they are satisfied, save it to a file named "Prompt-" followed by the current year, month, day, hour, minute, and second, separated by hyphens, followed by ".md".
Look up the current time with a shell command such as `date +%Y-%m-%d-%H-%M-%S`, and use the resulting digits in the filename.
For example, 14 October 2026 at 16:27:05 gives "Prompt-2026-10-14-16-27-05.md".
Never write the placeholder letters (YYYY, MM, DD, HH, mm for minutes, ss for seconds) into the filename.
Before saving, check with `test -e` whether a file with that name already exists.
Never overwrite an existing file: append a hyphen and the next unused number ("-1", "-2", …) before ".md".
