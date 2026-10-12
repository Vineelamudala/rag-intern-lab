# RAG Test Queries

> This file is the **evaluation set**. It lives outside `data/rag_docs/` on
> purpose and must **never** be ingested. The expected answers below come from
> the fictional training documents. Use them only to judge results; never
> hard-code them into the pipeline.

## How to use this file

1. Ingest all three strategies: `python scripts/ingest.py` (milestone M6).
2. Run every query on every strategy:
   - automatically (M11): `python scripts/evaluate.py --output eval_full.md`,
     or with `--retrieval-only` before the gateway is set up; or
   - by hand: `python scripts/ask.py "<question>" --strategy fixed --location Pune`,
     using the filters listed for each query.
   `scripts/evaluate.py` parses this file, so keep the `### Qnn: ... (CATEGORY)`
   headings and the `- **Field:**` lines intact.
3. Record the result in the table under each query:
   - **Top source(s) / section(s):** what retrieval returned (top 1–3).
   - **Retrieval hit?** `Y` if the expected source/section is in the top-k.
   - **Answer correct?** `Y` / `Partial` / `N` / `Correctly refused`.
   - **Notes:** for example "fixed chunk cut the table in half".
4. Summarise the hit rate and answer accuracy per strategy at the end.

Query categories: **S** = single document, **M** = multiple passages,
**F** = metadata filter, **D** = effective date or location rule,
**U** = unanswerable (must not be invented).

---

### Q01: Annual leave entitlement (S)

- **Question:** How many days of annual leave does a full-time employee get per year, and how does it accrue?
- **Filters:** none
- **Expected answer / key facts:** 18 days per leave year, accruing at 1.5 days per completed month.
- **Expected source:** `leave_policy.md` § Annual Leave > Entitlement
- **Expected behavior:** Answer with both facts and cite the leave policy.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q02: Hotel cap in a Tier-1 city (S)

- **Question:** What is the maximum hotel reimbursement per night when travelling to Bengaluru?
- **Filters:** none
- **Expected answer / key facts:** INR 7,000 per night. Bengaluru is a Tier-1 city.
- **Expected source:** `travel_expense_policy.md` § Accommodation
- **Expected behavior:** Combine the cap table with the Tier-1 city list and cite the travel policy.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q03: Maternity leave (S)

- **Question:** How much paid maternity leave is available, and who is eligible?
- **Filters:** none
- **Expected answer / key facts:** 26 weeks for each of the first two children, then 12 weeks. Eligibility requires at least 80 days worked in the 12 months before the expected delivery date.
- **Expected source:** `leave_policy.md` § Parental Leave > Maternity Leave
- **Expected behavior:** State both durations and the eligibility condition.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q04: Carry forward and lapse (M)

- **Question:** How many unused annual leave days can I carry forward, and what happens to the rest?
- **Filters:** none
- **Expected answer / key facts:** Up to 10 days carry forward. The excess lapses on 31 March.
- **Expected source:** `leave_policy.md` § Annual Leave > Carry Forward and Lapse; also `employee_faq.md` § Leave
- **Expected behavior:** Answer consistently from either or both passages. Both are acceptable citations.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q05: Remote employee travelling to a planning meeting (M)

- **Question:** I am a fully remote employee. Do I have to attend in-person meetings, and what daily allowance do I get when I travel to the Hyderabad office for one?
- **Filters:** none
- **Expected answer / key facts:** Fully remote employees must attend the quarterly planning meeting. Travel is covered by the Travel and Expense Policy. Hyderabad is Tier-1, so the per diem is INR 2,500 per day.
- **Expected source:** `remote_work_policy.md` § In-Person Events; `travel_expense_policy.md` § Per Diem / Travel for Remote Employees
- **Expected behavior:** Combine passages from two documents and cite both.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q06: Internet reimbursement (M)

- **Question:** How much internet reimbursement do hybrid and fully remote employees get, and how do they claim it?
- **Filters:** none
- **Expected answer / key facts:** INR 1,000 per month for hybrid and INR 1,500 per month for fully remote. Claim monthly in ExpensePro with the broadband bill.
- **Expected source:** `remote_work_policy.md` § Equipment and Allowances > Internet Reimbursement; `employee_faq.md` § Remote Work
- **Expected behavior:** Give both amounts and the claim process.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q07: New office address, circulars only (F)

- **Question:** What is the address of the new office?
- **Filters:** `--doc-type circular`
- **Expected answer / key facts:** Tower B, Orion Tech Park, Gachibowli, Hyderabad.
- **Expected source:** `hr_circular.md` § New Office Address
- **Expected behavior:** Retrieve only from the circular. Compare with the same query run without the filter.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q08: Expense claim deadline, Finance only (F)

- **Question:** What is the deadline for submitting expense claims, and what happens if I miss it?
- **Filters:** `--department Finance`
- **Expected answer / key facts:** Submit within 30 days of the trip end date. Claims more than 60 days late are not reimbursed without CFO approval.
- **Expected source:** `travel_expense_policy.md` § Receipts and Claim Submission
- **Expected behavior:** Every retrieved chunk is from the Finance-owned document.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q09: Pune anchor days (F, D)

- **Question:** Which days must hybrid employees be in the office?
- **Filters:** `--location Pune`
- **Expected answer / key facts:** For Pune, the anchor days are Monday and Wednesday. The default elsewhere is Tuesday and Thursday.
- **Expected source:** `remote_work_policy.md` § Anchor Days > Pune Office Anchor Days
- **Expected behavior:** Give the Pune rule. The Hyderabad circular must not appear in results because of the location filter.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q10: Remote work policy effective date (D)

- **Question:** From what date does the current remote and hybrid work policy apply, and what did it replace?
- **Filters:** none
- **Expected answer / key facts:** 1 July 2026 (version 2.0). It replaced the Flexible Work Guidelines issued in March 2024.
- **Expected source:** `remote_work_policy.md` § Purpose and Scope
- **Expected behavior:** Give the date and the replaced document.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q11: Ganesh Chaturthi in Hyderabad (D)

- **Question:** Is Ganesh Chaturthi a paid holiday for employees based in Hyderabad?
- **Filters:** none (also try `--location Hyderabad`)
- **Expected answer / key facts:** No. Ganesh Chaturthi (14 September 2026) is a Pune-only holiday. Hyderabad's location holidays are Makar Sankranti and Telangana Formation Day.
- **Expected source:** `holiday_calendar.md` § Location-Specific Holidays
- **Expected behavior:** Distinguish location-specific lists. Must not say "yes" just because the holiday appears in the calendar.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q12: Hyderabad relocation dates (D, F)

- **Question:** When do Hyderabad employees start reporting to the new campus, and where should they work during the move?
- **Filters:** `--location Hyderabad`
- **Expected answer / key facts:** All employees report from Tuesday, 1 September 2026. On 27–28 August 2026 they work from home. On 31 August only the IT and Administration teams report.
- **Expected source:** `hr_circular.md` § Summary / Move Schedule
- **Expected behavior:** Give the correct dates and cite the circular.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q13: Stock options (U)

- **Question:** What is the vesting schedule for employee stock options?
- **Filters:** none
- **Expected answer / key facts:** Not in the corpus.
- **Expected source:** none
- **Expected behavior:** Respond "I could not find the answer in the available documents." No invented vesting schedule and no citations.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q14: Singapore office leave (U)

- **Question:** How many days of annual leave do employees at the Singapore office get?
- **Filters:** none (also try `--location Singapore`)
- **Expected answer / key facts:** Not in the corpus. There is no Singapore office; the documents cover Hyderabad, Pune and Bengaluru only.
- **Expected behavior:** Refuse, or state that no Singapore-specific information exists. Must not apply the 18-day rule to Singapore as if it were sourced. With `--location Singapore`, retrieval returns only "All Locations" documents.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

### Q15: Gym membership (U)

- **Question:** What is the reimbursement limit for gym memberships?
- **Filters:** none
- **Expected answer / key facts:** Not in the corpus. Wellness support covers only the CareCircle counselling sessions, not gyms.
- **Expected source:** none
- **Expected behavior:** Respond "I could not find the answer in the available documents." Must not reuse unrelated limits such as the learning budget or client entertainment cap.

| Strategy | Top source(s) / section(s) | Retrieval hit? | Answer correct? | Notes |
| --- | --- | --- | --- | --- |
| fixed | | | | |
| recursive | | | | |
| structured | | | | |

---

## Summary (fill in)

| Strategy | Retrieval hit rate (Q01–Q12) | Answer accuracy (Q01–Q12) | Correct refusals (Q13–Q15) | Avg chunk size | Chunk count |
| --- | --- | --- | --- | --- | --- |
| fixed | /12 | /12 | /3 | | |
| recursive | /12 | /12 | /3 | | |
| structured | /12 | /12 | /3 | | |

**Chunking configuration used:** `CHUNK_SIZE=` `CHUNK_OVERLAP=` `TOP_K=` `MIN_SIMILARITY=`
**LLM provider and model:**
**Observations and recommendation:**
