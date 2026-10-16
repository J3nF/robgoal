# ToDos

Goal: Produce a prompt for an agentic LLM to get the following points done:

1. Change the format of @docs/lab-book.md to be
    - Human-readable (especially skim-reading and scanning for particular runs should be possible).
    - Have a simple numerical run ID (i.e., scan 1, scan 2, etc.), a time stamp, and one or two keywords as names of the run (and, thus, folder names).
      In order to have the IDs sorted correctly, the ID should start with a leading 0 (i.e., 01, 02, etc).
  Research if there is some format naturally fulfilling these wishes. Maybe it is JSON?
  Keep in mind that lab-book.md is to basically be a dictionary / reference of when and what something was done, as well as why.
2.