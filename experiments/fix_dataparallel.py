"""
直接修复DataParallel问题的脚本
确保twostep_executor.py正确处理DataParallel包装的模型
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
veccity_path = os.path.join(project_root, 'VecCity-main')
executor_path = os.path.join(veccity_path, 'veccity/executor/twostep_executor.py')

print("="*80)
print("FIXING DATAPARALLEL ISSUE IN TWOSTEP_EXECUTOR")
print("="*80)
print(f"Target file: {executor_path}")
print()

# Read the file
with open(executor_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Backup if not exists
backup_path = executor_path + '.backup_dataparallel_fix'
if not os.path.exists(backup_path):
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"✓ Backup saved: {backup_path}")

# Find and fix the issues
modified = False
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]

    # Fix 1: Add _get_model helper method after __init__
    if 'self.cache_dir = ' in line and '_get_model' not in ''.join(lines):
        # Find the end of __init__ method (next blank line or next def)
        j = i + 1
        while j < len(lines) and not (lines[j].strip() == '' or lines[j].strip().startswith('def ')):
            j += 1

        # Insert _get_model method before the next method
        helper_method = '''
    def _get_model(self):
        """Helper to get the actual model (unwrap DataParallel if needed)"""
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

'''
        new_lines.append(line)
        # Add remaining lines of __init__
        for k in range(i+1, j):
            new_lines.append(lines[k])
        new_lines.append(helper_method)
        i = j
        modified = True
        print("✓ Added _get_model() helper method")
        continue

    # Fix 2: Replace self.model.run with self._get_model().run
    if 'return self.model.run(' in line:
        new_lines.append(line.replace('self.model.run(', 'self._get_model().run('))
        modified = True
        print("✓ Fixed train() method to use _get_model()")
        i += 1
        continue

    # Fix 3: Replace self.model.load_state_dict with self._get_model().load_state_dict
    if 'self.model.load_state_dict(' in line:
        new_lines.append(line.replace('self.model.load_state_dict(', 'self._get_model().load_state_dict('))
        modified = True
        print("✓ Fixed load_model() to use _get_model()")
        i += 1
        continue

    # Fix 4: Replace self.model.optimizer with self._get_model().optimizer
    if 'self.model.optimizer' in line and 'self._get_model()' not in line:
        new_lines.append(line.replace('self.model.optimizer', 'self._get_model().optimizer'))
        modified = True
        print("✓ Fixed optimizer access to use _get_model()")
        i += 1
        continue

    # Fix 5: Replace self.model.state_dict with self._get_model().state_dict in save_model
    if "config['model_state_dict'] = self.model.state_dict()" in line:
        new_lines.append(line.replace('self.model.state_dict()', 'self._get_model().state_dict()'))
        modified = True
        print("✓ Fixed save_model() to use _get_model()")
        i += 1
        continue

    # Keep the line as-is
    new_lines.append(line)
    i += 1

# Write back if modified
if modified:
    with open(executor_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print()
    print("="*80)
    print("✅ DATAPARALLEL FIX APPLIED SUCCESSFULLY")
    print("="*80)
    print()
    print("The executor can now handle both:")
    print("  - Regular models: model.run()")
    print("  - DataParallel wrapped models: model.module.run()")
    print()
else:
    print()
    print("="*80)
    print("ℹ️  NO CHANGES NEEDED - File already fixed")
    print("="*80)
    print()
