from pathlib import Path
import pymupdf


OUTPUT_DIR = Path("data/documents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


documents = {
    "employee_handbook.pdf": {
        "title": "NexaTech Solutions - Employee Handbook",
        "sections": [
            (
                "1. Working Hours",
                "NexaTech Solutions follows a standard working schedule of 9:00 AM to "
                "6:00 PM from Monday to Friday. Employees are expected to maintain "
                "professional attendance and communicate planned absences with their "
                "manager."
            ),
            (
                "2. Employee Eligibility",
                "All full-time employees who have completed their probation period "
                "are eligible for the standard employee benefits described in this handbook."
            ),
            (
                "3. Code of Conduct",
                "Employees are expected to maintain professional behavior, respect "
                "colleagues, protect company information, and follow all applicable "
                "company policies."
            ),
            (
                "4. Information Security",
                "Employees must protect company credentials and confidential information. "
                "Passwords must not be shared with other employees or external parties."
            ),
            (
                "5. Performance Reviews",
                "Employees participate in formal performance reviews twice each year. "
                "Managers evaluate performance against role expectations, objectives, "
                "team contribution, and professional development."
            ),
        ],
    },

    "work_from_home_policy.pdf": {
        "title": "NexaTech Solutions - Work From Home Policy",
        "sections": [
            (
                "Policy ID: HR-WFH-008",
                "This policy defines the eligibility requirements and procedures for "
                "employees requesting remote work."
            ),
            (
                "1. Eligibility",
                "Full-time employees who have completed at least one year of continuous "
                "service are eligible for regular work-from-home arrangements."
            ),
            (
                "2. Monthly Remote Work Allowance",
                "Eligible employees may work from home for up to 8 working days per "
                "calendar month. Unused remote work days cannot be carried forward to "
                "the following month."
            ),
            (
                "3. Approval Process",
                "Employees must submit work-from-home requests through the employee "
                "portal at least one working day before the requested date. The "
                "employee's reporting manager must approve the request."
            ),
            (
                "4. Exceptions",
                "Employees may request temporary remote work beyond the normal monthly "
                "allowance during exceptional circumstances. Such requests require "
                "manager approval and may require HR approval."
            ),
            (
                "5. Remote Work Responsibilities",
                "Employees working remotely must maintain normal working hours, remain "
                "available through approved communication channels, and protect company "
                "data and equipment."
            ),
        ],
    },

    "leave_policy.pdf": {
        "title": "NexaTech Solutions - Leave Policy",
        "sections": [
            (
                "Policy ID: HR-LEAVE-004",
                "This policy defines the types of leave available to eligible employees "
                "and the procedures for requesting leave."
            ),
            (
                "1. Annual Leave",
                "Eligible full-time employees receive 24 days of annual paid leave "
                "per calendar year. Annual leave should normally be requested in advance."
            ),
            (
                "2. Sick Leave",
                "Employees may use sick leave when they are unable to work because of "
                "illness or a medical condition. Supporting documentation may be "
                "requested for extended periods of absence."
            ),
            (
                "3. Casual Leave",
                "Employees may request casual leave for short-term personal matters. "
                "Approval depends on operational requirements and manager availability."
            ),
            (
                "4. Carry Forward",
                "Up to 10 unused annual leave days may be carried forward into the "
                "following calendar year. Other leave categories generally cannot "
                "be carried forward."
            ),
            (
                "5. Leave Approval",
                "Leave requests must be submitted through the employee portal and "
                "approved by the employee's reporting manager."
            ),
        ],
    },

    "it_security_policy.pdf": {
        "title": "NexaTech Solutions - IT Security Policy",
        "sections": [
            (
                "Policy ID: IT-SEC-012",
                "This policy establishes security requirements for protecting company "
                "systems, accounts, devices, and confidential information."
            ),
            (
                "1. Password Requirements",
                "Employees must use strong passwords containing a combination of "
                "uppercase letters, lowercase letters, numbers, and special characters. "
                "Passwords must not be reused across company systems."
            ),
            (
                "2. Multi-Factor Authentication",
                "Multi-factor authentication is required for all critical company "
                "systems and externally accessible business applications."
            ),
            (
                "3. Company Devices",
                "Employees must use company-approved security software and keep operating "
                "systems and applications updated. Lost or stolen company devices must "
                "be reported immediately to the IT security team."
            ),
            (
                "4. Data Protection",
                "Confidential company information must only be stored using approved "
                "company systems. Employees must not upload confidential information "
                "to unauthorized external services."
            ),
            (
                "5. Security Incidents",
                "Suspected phishing attacks, unauthorized access, malware infections, "
                "or accidental data disclosure must be reported to the IT security "
                "team immediately."
            ),
        ],
    },

    "reimbursement_policy.pdf": {
        "title": "NexaTech Solutions - Employee Reimbursement Policy",
        "sections": [
            (
                "Policy ID: FIN-EXP-006",
                "This policy defines the expenses that employees may claim for legitimate "
                "business activities."
            ),
            (
                "1. Business Travel",
                "Reasonable transportation and accommodation expenses incurred during "
                "approved business travel may be reimbursed."
            ),
            (
                "2. Meals",
                "Employees may claim reasonable meal expenses during approved business "
                "travel. Original receipts must be submitted with the reimbursement claim."
            ),
            (
                "3. Internet Reimbursement",
                "Eligible employees working under approved remote arrangements may claim "
                "up to INR 1,000 per month for business-related internet expenses."
            ),
            (
                "4. Approval Limits",
                "Expense claims must be approved by the employee's reporting manager. "
                "Higher-value claims may require additional finance approval."
            ),
            (
                "5. Submission Deadline",
                "Reimbursement claims should normally be submitted within 30 days of "
                "the expense date."
            ),
        ],
    },

    "employee_benefits.pdf": {
        "title": "NexaTech Solutions - Employee Benefits Guide",
        "sections": [
            (
                "1. Health Insurance",
                "Eligible full-time employees receive company-sponsored health insurance "
                "coverage according to the applicable benefits plan."
            ),
            (
                "2. Professional Development",
                "Employees may receive financial support for approved professional courses, "
                "certifications, and training programs related to their role."
            ),
            (
                "3. Performance Bonus",
                "Eligible employees may receive performance-based bonuses according to "
                "company performance, individual objectives, and applicable compensation "
                "guidelines."
            ),
            (
                "4. Employee Assistance",
                "The company provides access to employee support resources covering "
                "professional development and workplace assistance."
            ),
            (
                "5. Benefits Eligibility",
                "Specific benefit eligibility depends on employment type, length of service, "
                "and the terms of the applicable benefits plan."
            ),
        ],
    },
        "travel_expense_policy.pdf": {
        "title": "NexaTech Solutions - Travel Expense Policy",
        "sections": [
            (
                "Policy ID: FIN-TRAVEL-009",
                "This policy defines reimbursement rules for approved "
                "business travel undertaken by employees."
            ),
            (
                "1. Travel Authorization",
                "Employees must obtain manager approval before making "
                "business travel arrangements."
            ),
            (
                "2. Transportation",
                "Reasonable airfare, rail, taxi, and other approved "
                "transportation expenses may be reimbursed when supported "
                "by valid receipts."
            ),
            (
                "3. Accommodation",
                "Employees may claim reasonable hotel accommodation costs "
                "for approved business trips. Accommodation should follow "
                "company travel guidelines."
            ),
            (
                "4. Travel Meals",
                "Reasonable meal expenses incurred during approved business "
                "travel may be claimed with original receipts."
            ),
            (
                "5. Travel Claims",
                "Travel expense claims should be submitted within 30 days "
                "after completion of the business trip."
            ),
        ],
    },

    "device_usage_policy.pdf": {
        "title": "NexaTech Solutions - Company Device Usage Policy",
        "sections": [
            (
                "Policy ID: IT-DEVICE-015",
                "This policy defines acceptable use and security requirements "
                "for company-provided laptops, mobile devices, and equipment."
            ),
            (
                "1. Approved Devices",
                "Employees must use company-approved devices for accessing "
                "confidential business systems and information."
            ),
            (
                "2. Software Installation",
                "Employees must not install unauthorized software or applications "
                "on company-managed devices."
            ),
            (
                "3. Device Security",
                "Company devices must use approved security controls and remain "
                "protected with current operating system and security updates."
            ),
            (
                "4. Remote Usage",
                "Employees working remotely must protect company devices from "
                "unauthorized access and use secure network connections."
            ),
            (
                "5. Lost Devices",
                "Lost or stolen company devices must be reported immediately "
                "to the IT security team."
            ),
        ],
    },

    "attendance_policy.pdf": {
        "title": "NexaTech Solutions - Attendance and Working Hours Policy",
        "sections": [
            (
                "Policy ID: HR-ATT-011",
                "This policy defines standard working hours, attendance "
                "expectations, and absence communication requirements."
            ),
            (
                "1. Standard Working Hours",
                "The standard working schedule is 9:00 AM to 6:00 PM, "
                "Monday through Friday."
            ),
            (
                "2. Attendance",
                "Employees are expected to maintain professional attendance "
                "and remain available during scheduled working hours."
            ),
            (
                "3. Planned Absence",
                "Employees should communicate planned absences to their "
                "reporting manager in advance."
            ),
            (
                "4. Remote Employees",
                "Employees working remotely must maintain their normal "
                "working schedule and remain available through approved "
                "communication channels."
            ),
            (
                "5. Attendance Records",
                "Employees may be required to maintain accurate attendance "
                "records according to applicable company procedures."
            ),
        ],
    },

    "data_privacy_policy.pdf": {
        "title": "NexaTech Solutions - Data Privacy Policy",
        "sections": [
            (
                "Policy ID: IT-PRIV-018",
                "This policy establishes requirements for protecting personal "
                "and confidential information handled by employees."
            ),
            (
                "1. Personal Information",
                "Employees must handle personal information only for legitimate "
                "business purposes and according to company procedures."
            ),
            (
                "2. Access Control",
                "Access to confidential and personal information must be limited "
                "to employees who require it for their assigned responsibilities."
            ),
            (
                "3. Data Sharing",
                "Employees must not share confidential or personal information "
                "with unauthorized individuals or external parties."
            ),
            (
                "4. Data Storage",
                "Company information must be stored using approved company "
                "systems and authorized storage locations."
            ),
            (
                "5. Privacy Incidents",
                "Suspected unauthorized disclosure, loss, or misuse of personal "
                "information must be reported to the appropriate security team."
            ),
        ],
    },
}


def create_pdf(filename, title, sections):
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text(
        (72, 72),
        title,
        fontsize=20,
        fontname="helv",
    )

    y = 110

    for heading, text in sections:
        if y > 700:
            page = doc.new_page()
            y = 72

        page.insert_text(
            (72, y),
            heading,
            fontsize=14,
            fontname="helv",
        )

        y += 25

        rect = pymupdf.Rect(72, y, 520, y + 100)

        page.insert_textbox(
            rect,
            text,
            fontsize=10,
            fontname="helv",
            lineheight=1.4,
        )

        y += 125

    output_path = OUTPUT_DIR / filename
    doc.save(output_path)
    doc.close()

    print(f"Created: {output_path}")


for filename, document in documents.items():
    create_pdf(
        filename,
        document["title"],
        document["sections"],
    )

print("\nAll enterprise documents created successfully!")