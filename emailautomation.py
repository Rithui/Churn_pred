import imaplib  # Using standard imaplib instead of imaplib2
import email
import time
import requests
import json
import os
from email.header import decode_header
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_automation.log"),
        logging.StreamHandler()
    ]
)

# Configuration
EMAIL = "appugamerz2003@gmail.com"  # Replace with your email
PASSWORD = "odyg gmez psdu oibg"  # Replace with your app password (for Gmail)
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993
CHECK_INTERVAL = 10  # seconds
API_ENDPOINT = "http://127.0.0.1:5000/api/predict"

# Create a file to store processed email IDs
if not os.path.exists('processed_emails.txt'):
    with open('processed_emails.txt', 'w') as f:
        f.write('')

def get_processed_emails():
    """Read the list of already processed email IDs"""
    try:
        with open('processed_emails.txt', 'r') as f:
            return set(f.read().splitlines())
    except Exception as e:
        logging.error(f"Error reading processed emails: {e}")
        return set()

def add_processed_email(email_id):
    """Add an email ID to the processed list"""
    try:
        with open('processed_emails.txt', 'a') as f:
            f.write(f"{email_id}\n")
    except Exception as e:
        logging.error(f"Error adding processed email: {e}")

def clean_text(text):
    """Clean and decode email text"""
    if isinstance(text, bytes):
        text = text.decode()
    return text

def decode_email_subject(subject):
    """Decode email subject"""
    if subject is None:
        return ""
    decoded_header = decode_header(subject)
    subject_parts = []
    for content, encoding in decoded_header:
        if isinstance(content, bytes):
            if encoding:
                content = content.decode(encoding)
            else:
                content = content.decode('utf-8', errors='replace')
        subject_parts.append(content)
    return " ".join(subject_parts)

def get_email_body(msg):
    """Extract the email body text"""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            # Get text content
            if content_type == "text/plain":
                try:
                    body = part.get_payload(decode=True)
                    return clean_text(body)
                except:
                    pass
    else:
        # If not multipart, just get the payload
        try:
            body = msg.get_payload(decode=True)
            return clean_text(body)
        except:
            pass
    
    return ""

def classify_email(email_text):
    """Send email text to the Flask API for classification"""
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"email_text": email_text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error classifying email: {e}")
        return None

def check_emails():
    """Check for new emails and classify them"""
    try:
        # Connect to the IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL, PASSWORD)
        mail.select("INBOX")
        
        # Get all emails from inbox
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()
        
        # Get already processed emails
        processed_emails = get_processed_emails()
        
        # Process new emails (starting from the most recent)
        for email_id in reversed(email_ids):
            email_id_str = email_id.decode()
            
            # Skip already processed emails
            if email_id_str in processed_emails:
                continue
                
            # Fetch the email
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            
            # Get email details
            subject = decode_email_subject(msg["Subject"])
            from_address = msg.get("From", "")
            date_str = msg.get("Date", "")
            
            # Get email body
            body = get_email_body(msg)
            
            # Combine subject and body for better classification
            full_text = f"Subject: {subject}\n\n{body}"
            
            # Classify the email
            result = classify_email(full_text)
            
            if result:
                classification = result["result"]
                confidence = result["confidence"]
                
                # Log the result
                logging.info(f"New Email: {subject}")
                logging.info(f"From: {from_address}")
                logging.info(f"Classification: {classification} (Confidence: {confidence}%)")
                logging.info("-" * 50)
                
                # Mark as processed
                add_processed_email(email_id_str)
        
        # Logout
        mail.logout()
        
    except Exception as e:
        logging.error(f"Error checking emails: {e}")

def main():
    """Main function to run the email checking loop"""
    logging.info("Starting email automation service")
    logging.info(f"Checking for new emails every {CHECK_INTERVAL} seconds")
    logging.info("-" * 50)
    
    while True:
        try:
            check_emails()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logging.info("Email automation service stopped")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()