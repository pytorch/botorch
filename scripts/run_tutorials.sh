git config --global user.email "santorella@meta.com"
git config --global user.name "Elizabeth Santorella"
echo "cloning"
git clone https://github.com/pytorch/botorch.git botorch-main
cd botorch-main
echo "creating file"
touch test_file.csv
echo "Checking out branch artifacts"
git fetch origin artifacts
git checkout artifacts
git add test_file.csv
echo "Committing"
git commit test_file.csv -m "Adding most recent tutorials output"
echo "Pushing"
git push origin artifacts
# clean up
cd ..
rm -rf botorch-main 