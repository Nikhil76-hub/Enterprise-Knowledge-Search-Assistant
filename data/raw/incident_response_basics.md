# Acme Corp Information Security Incident Response Guide

## Incident Severity Classifications and SLAs
Security incidents are classified into three distinct tiers. Severity 1 (Critical) involves active ransomware, credential theft affecting production infrastructure, or confirmed data exfiltration, requiring security response within 15 minutes. Severity 2 (High) covers compromised non-critical servers or active phishing campaigns with clicked payloads, carrying a 60-minute response SLA. Severity 3 (Medium/Low) includes lost encrypted hardware, suspicious spam, or minor policy breaches, with a 4-hour SLA.

## Immediate Containment Actions
If you suspect your computer or server has been compromised, take immediate containment steps: immediately disconnect the Ethernet cable and switch off Wi-Fi. Under no circumstances should you power down, reboot, or close open applications, as volatile system RAM contains vital forensic evidence. Do not attempt manual cleanup or run unauthorized antivirus tools before the Security Operations Center (SOC) gives permission.

## Reporting Hotlines and War Room Protocol
Report all potential security anomalies immediately to the 24/7 SOC Emergency Hotline at +1-800-555-0911 or email security-alert@acmecorp.com. For Severity 1 and Severity 2 incidents, the Incident Commander establishes a dedicated Zoom war room at zoom.us/my/acmesoc and creates a real-time tracking Slack channel named #incident-[YYYYMMDD]-[incident-id]. Updates are communicated to company leadership every 30 minutes during active outages.

