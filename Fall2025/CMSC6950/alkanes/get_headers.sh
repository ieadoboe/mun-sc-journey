for file in *.pdb
do
	head -n 2 $file >> headers.txt
done
