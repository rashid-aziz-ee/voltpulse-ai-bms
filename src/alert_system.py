import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime

def send_alert(sender_email, app_password, target_email, cell_id, voltage, temp, risk_level):
    """
    Sends an automated email alert when a cell is critical.
    """
    if not sender_email or not app_password:
        print("⚠️ Email credentials not provided. Simulating alert.")
        return False

    try:
        subject = f"🚨 EMERGENCY: Thermal Runaway Risk on {cell_id}"
        
        body = f"""
        VoltPulse-AI Autonomous Alert System
        ====================================
        Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        CRITICAL ALERT DETECTED IN 132kV BATTERY BANK
        
        Cell ID: {cell_id}
        Risk Level: {'CRITICAL (Fire Risk)' if risk_level == 2 else 'WARNING'}
        Live Voltage: {voltage} V
        Live Temperature: {temp} °C
        
        ACTION TAKEN BY AI:
        - Relays tripped for {cell_id} isolation.
        - Load shedding protocol activated.
        
        Please dispatch maintenance immediately.
        
        - VoltPulse-AI BMS
        """
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = target_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        text = msg.as_string()
        server.sendmail(sender_email, target_email, text)
        server.quit()
        
        print(f"✅ Alert email successfully sent to {target_email}!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email alert: {str(e)}")
        return False
