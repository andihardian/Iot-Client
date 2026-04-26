import requests
import time

API_URL      = "http://127.0.0.1:8000/api/door/unlock"
DEVICE_TOKEN = "raspi-token-001"

TEST_CASES = [
    {"identifier": "RFID-ABC123",  "method": "rfid",  "label": "Kartu RFID valid"},
    {"identifier": "RFID-HACKER",  "method": "rfid",  "label": "Kartu tidak dikenal"},
    {"identifier": "PIN-1234",     "method": "pin",   "label": "PIN masuk"},
    {"identifier": "FACE-user_1",  "method": "face",  "label": "Face recognition dikenal"},
    {"identifier": "FACE-UNKNOWN", "method": "face",  "label": "Wajah tidak dikenal"},
]

def run_test(case):
    print(f"\n[TEST] {case['label']}")
    print(f"       ID     : {case['identifier']}")
    print(f"       Method : {case['method']}")
    try:
        res  = requests.post(API_URL, json={
            "device_token": DEVICE_TOKEN,
            "identifier":   case["identifier"],
            "method":       case["method"],
        }, timeout=5)
        data = res.json()
        icon = "✅" if data['status'] == 'granted' else "❌"
        print(f"       Status : {icon} {data['status'].upper()}")
        print(f"       Pesan  : {data['message']}")
        return data['status']
    except requests.exceptions.ConnectionError:
        print("       [ERROR] Laravel tidak jalan!")
        return None
    except Exception as e:
        print(f"       [ERROR] {e}")
        return None

if __name__ == "__main__":
    print("="*50)
    print("  SMART DOOR — Auto Simulator Test")
    print("="*50)

    granted = denied = errors = 0

    for case in TEST_CASES:
        result = run_test(case)
        if result == 'granted':   granted += 1
        elif result == 'denied':  denied  += 1
        else:                     errors  += 1
        time.sleep(1)

    print("\n" + "="*50)
    print("  HASIL TEST")
    print("="*50)
    print(f"  ✅ Granted : {granted}")
    print(f"  ❌ Denied  : {denied}")
    print(f"  ⚠️  Error   : {errors}")
    print(f"  📊 Total   : {len(TEST_CASES)}")
    print("="*50)