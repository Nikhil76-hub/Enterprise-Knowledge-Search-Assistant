# Acme Corp IT Password and VPN Access Guide

## Password Complexity and Lifecycle Requirements
All corporate Acme Corp domain accounts must adhere to strict password complexity standards. Passwords must be at least 14 characters in length and include at least one uppercase letter, one lowercase letter, one number, and one special character (!@#$%^&*). Passwords automatically expire every 90 days. The system remembers your previous 6 passwords, preventing reuse. Accounts are locked for 30 minutes after 5 consecutive incorrect login attempts.

## Multi-Factor Authentication Setup
Multi-factor authentication (MFA) via Okta Verify is mandatory across all Acme enterprise applications. SMS-based authentication is restricted and only permitted as an emergency fallback by exception. New hires must configure Okta Verify push notifications on their corporate or personal mobile device within 24 hours of account activation. If you lose your registered MFA device, you must verify your identity via video call with IT support.

## GlobalProtect VPN Configuration and Support
Remote access to internal servers, code repositories, and databases requires an active connection via Palo Alto GlobalProtect VPN. Open the GlobalProtect client, set the portal address to vpn.acmecorp.com:443, and authenticate using your Okta credentials. Sessions automatically terminate after 8 hours of continuous connection or 30 minutes of idle time. For account unlocks or connection diagnostics, call the IT Service Desk at +1-800-555-0199 or email it-servicedesk@acmecorp.com.

