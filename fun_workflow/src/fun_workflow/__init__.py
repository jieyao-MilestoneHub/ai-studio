"""fun_workflow: the LINE group's playground, built on ai-studio.

Everything between a message in a group chat and a rendered file back in
it: the webhook (`api`), the LINE clients and trigger parsing (`bots`), the
SQLite queue, worker loop and renderers (`pipeline`), the questions and
persona the models are given (`prompts`), and the `funapp` command line
(`cli`) that wires all of it to ai-studio's pod. See README.md.
"""
