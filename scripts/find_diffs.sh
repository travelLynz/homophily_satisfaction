in_files=`ls $1/*.txt`
for eachfile in $in_files
do
  echo "Processing diff: $eachfile"
  diff $eachfile $2/$(basename $eachfile)
done
