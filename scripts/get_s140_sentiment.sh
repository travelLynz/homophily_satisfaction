# #!/bin/bash
# IN_FILES= "../Data/s140/in/"*
# for f in $IN_FILES
#   do
#     echo "Processing $f file..."
#     # take action on each file. $f store current file name
#     # out = $OUT_FILES + 'out_' + $f
#     # print "curl --data-binary @" + $f + '"http://www.sentiment140.com/api/bulkClassify?query=accommodation" -o ' + $out
#   done
in_files=`ls $1/in/*.txt`
out_dir=" $1/out/"
for eachfile in $in_files
do
   echo $eachfile
   echo $out_dir'out_'$(basename $eachfile)
   curl --data-binary @$eachfile "http://www.sentiment140.com/api/bulkClassify?query=accommodation" -o $out_dir'out_'$(basename $eachfile)
done
