project_root=`git rev-parse --show-toplevel`
modified_files=`git diff --name-only | grep -E '\.py$' | sed 's,^,'"$project_root"'/,' | xargs ls -d 2> /dev/null`

python -m isort $modified_files
python -m black --line-length=88 $modified_files
flake8 $modified_files

modified_files=`git diff --staged --name-only | grep -E '\.py$' | sed 's,^,'"$project_root"'/,' | xargs ls -d 2> /dev/null`

python -m isort $modified_files
python -m black --line-length=88 $modified_files
flake8 $modified_files

read -p "Press enter to continue"