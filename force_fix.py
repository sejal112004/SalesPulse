
import os

path = r"c:\Users\sejal\OneDrive\Desktop\SalesPulse\analyzer\templates\create_dashboard.html"

print(f"Target path: {path}")

try:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Original content length: {len(content)}")
    if "col==date_col" in content:
        print("Confirmed: 'col==date_col' exists in file.")
    else:
        print("Weird: 'col==date_col' NOT found in file.")

    # Perform replacement
    new_content = content.replace("col==date_col", "col == date_col")
    new_content = new_content.replace("col=='Revenue'", "col == 'Revenue'")

    if content == new_content:
        print("No changes changed. (Maybe already fixed?)")
    else:
        print("Applying changes...")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())
        print("Write completed and flushed.")

    # Verify immediately
    with open(path, 'r', encoding='utf-8') as f:
        final_content = f.read()
        print(f"New content length: {len(final_content)}")
        if "col == date_col" in final_content:
            print("Verification SUCCESS: 'col == date_col' FOUND.")
        else:
            print("Verification FAILED: 'col == date_col' NOT FOUND.")
        
        if "col==date_col" in final_content:
            print("Verification FAILED: Old 'col==date_col' STILL PRESENT.")
        else:
             print("Verification SUCCESS: Old 'col==date_col' GONE.")

except Exception as e:
    print(f"Error: {e}")
