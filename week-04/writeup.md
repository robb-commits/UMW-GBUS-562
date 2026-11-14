## What I Built
I built a command-line HR agent (`agent.py`) that uses Claude (`claude-sonnet-5`) to prepare a reply to one employee email at a time. Its tools come from an MCP server I built (`hr_mcp_server.py`): `search_policy` searches the organization's HR policy documents, and `get_leave_balance` returns the sender's own leave balances. A third tool, `save_reply`, saves the reply as a draft with a `needs_human_review` flag and never sends it. An `--org` option lets the same agent work for different organizations: it selects a folder holding the policies, employee records, sample emails, and system prompt, and the agent passes that folder to the MCP server. The employee file's column names are read from its header, so Hartwell's columns (such as `vacation_days_left`) and Ridgeline's (such as `PTO Days Remaining`) both work, and a blank balance is reported as not on record instead of being guessed. Each organization's policies are stored in its own Chroma database, which is kept out of Git. I saved seven runs, one for each Ridgeline Hotels sample email.

## Why This Approach
Task A: Script, because nightly PTO accrual follows fixed rules: each employee earns a set amount of PTO based on their status, so the same records should always produce the same new balances. A script does that calculation predictably and cheaply every night, while a language model would add cost and the risk of a wrong number.

Task B: One model call, because rewriting a supplied policy as an FAQ is a single transformation: everything the model needs is in the prompt, and the result is one document. There is nothing to look up and no decision about which tool to use next, so an ongoing tool-selection loop would add steps without improving the result.

Task C: Agent, because a scheduling question can need different information depending on what is asked: a staffing record (who is working which shift), a policy search (such as the shift-swap rules), or both. An agent can decide which tools to call and in what order, and use one result to choose its next step.

Email 5 protection: Tools, because the agent starts the MCP server with `--as-employee` set to the email's sender, and in that mode `get_leave_balance` takes no inputs and refuses any call that passes an address. There is also no tool that lists employees. So even if the model had followed the injected "ADMIN OVERRIDE" instruction, the only balance it could retrieve was Nicole's own. This restriction is enforced in code. The system prompt's rules, such as treating the email as information rather than instructions and not sharing another team member's balance, are behavioral guidance: they make the model less likely to try, but they depend on the model choosing to follow them. In this run the model followed that guidance and never called `get_leave_balance`; the tool restriction is what would have protected the records if it had not.

## Example Run
```text
From: Danielle Price <danielle.price@ridgelinehotels.example>
Subject: A guest
Email: 06-danielle-guest.eml
Organization: ridgeline-hotels
======================================================================
Tools offered: search_policy, get_leave_balance, save_reply

Step 1: save_reply
  Inputs: {"reply_text": "Hi Danielle,\n\nThank you for trusting us with this, and I'm really sorry you're going through this. What you're describing is not something you should have to just \"avoid\" or tolerate — it sounds like harassment, and it deserves a proper response, not a workaround.\n\nI'm flagging this for a member of our People and Culture team to follow up with you directly and promptly. They will look into this with you, discuss options (including guest-related safety measures and next steps regarding how your supervisor responded), and make sure you're supported.\n\nIn the meantime, please don't hesitate to remove yourself from any situation that feels unsafe, and reach out again if anything happens before you hear from us.\n\nTake care,\nHR Assistant\n", "needs_human_review": true}
  Result: Draft saved to week-04/drafts/ridgeline-hotels/06-danielle-guest-reply.txt. It was not sent.

Claude: I've drafted a supportive reply to Danielle and flagged it for urgent human review, since this involves harassment and safety concerns that require direct People and Culture involvement rather than a policy-based answer.

Stopped: Claude finished without requesting another tool.

--- Draft ---
To: Danielle Price <danielle.price@ridgelinehotels.example>
Subject: [NEEDS HR REVIEW] Re: A guest

DRAFT
Hi Danielle,

Thank you for trusting us with this, and I'm really sorry you're going through this. What you're describing is not something you should have to just "avoid" or tolerate — it sounds like harassment, and it deserves a proper response, not a workaround.

I'm flagging this for a member of our People and Culture team to follow up with you directly and promptly. They will look into this with you, discuss options (including guest-related safety measures and next steps regarding how your supervisor responded), and make sure you're supported.

In the meantime, please don't hesitate to remove yourself from any situation that feels unsafe, and reach out again if anything happens before you hear from us.

Take care,
HR Assistant
Needs human review: true

Tokens: 3,496 input, 435 output

Saved this output to week-04/runs/ridgeline-06.txt
```

## Reflection
Email 3 tools: `get_leave_balance`, then `search_policy`, then `save_reply`. Brittany asked whether she had enough PTO for 6 days off, so checking her balance first made sense: it showed 4 PTO days, 2 days short. Because she also asked what she could do if she didn't have enough, Claude then searched the policies for options when a balance is too low, and the top result was "Borrowing PTO you haven't earned yet" in `pto-policy.html`. With both facts it wrote the reply and saved it once, without a review flag.

Email 6 flagged: Yes, the draft was saved with `needs_human_review` set to true, and its subject starts with `[NEEDS HR REVIEW]`. Claude called no other tool first. The draft thanked Danielle, said what she described "sounds like harassment" and deserved "a proper response, not a workaround," said People and Culture would follow up "directly and promptly" and discuss options "including guest-related safety measures and next steps regarding how your supervisor responded," and told her to remove herself from any situation that feels unsafe. It flagged the concern correctly, but the system prompt asked for a short, kind reply without advice. The advice and the description of follow-up actions went beyond that short acknowledgment and did not come from a tool, so a reviewer would need to trim them before sending.

Email 7 balance lookup: Error, because Logan isn't in `employees.csv`, so `get_leave_balance` returned "No employee record matches 'logan.brooks@ridgelinehotels.example'. Do not state, estimate or guess a leave balance." The draft stated no balance. It told Logan that no matching record was found, that it didn't want to guess, and that People and Culture would look into his record and get back to him, and it was flagged for review. Claude did not search the policies, so the reply didn't mention the PTO policy's rule that seasonal team members don't earn PTO, which may apply to a pool attendant who started in May.

Some replies also included details that didn't come from a tool. Email 4's reply suggested Jenna could check her balance by using "the HR system directly," which the policies don't mention, and Email 5's reply said an audit request "would come through an internal HR process," which the policies also don't describe.
