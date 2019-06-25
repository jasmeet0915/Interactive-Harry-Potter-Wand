echo ""
echo ".... Script to Rename Images ...."

count=1
# variable to store directory name
directory_name="A_testing2"
# change the value of the variable to required directory

for img in $directory_name/*;
do
	echo ""
	echo ">>> Renaming File " $img " to " $count".jpg"

	# full path of images has to be specified in mv command
	# as they are not in currently activated directory

	mv $img /home/jasmeet/PycharmProjects/Interactive_Harry_Potter_Wand/$directory_name/$count.jpg
	let count++
done
