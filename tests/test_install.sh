set -e

ENV_DIR="../p3"
virtualenv -p python3 "$ENV_DIR"

#activate environment
. "$ENV_DIR/bin/activate"

#install needed packages:
pip install cython
pip install numpy # needed for testing
pip install https://github.com/realead/uttemplate/zipball/master # needed for testing


if [ "$1" = "from-github" ]; then
    echo "Installing setup.py from github..."
    pip install https://github.com/realead/precsum/zipball/master
else
    echo "Installing local setup.py..."
    for dir_name in "build" ".eggs" "dist"
    do
        if [ -d "../$dir_name" ]; then
           echo "clean build, deleting ../$dir_name directory"
           rm -r "../$dir_name"
        fi; 
    done  
    (cd .. && python setup.py install)
fi;

echo "Installed packages:"
pip freeze

#tests:
sh run_unit_tests.sh

echo "\nprecision performance tests running..."
python precision_performance_1d.py | tee tmp_result_precision_1d.txt

echo "\ntimeit performance tests running..."
echo "  1d performance tests..."
python timeit_performance_1d.py | tee tmp_result_timeit_1d.txt
echo "  2d performance tests..."
python timeit_performance.py | tee tmp_result_timeit.txt

#clean or keep the environment
if [ "$2" = "keep" ]; then
   echo "keeping enviroment $ENV_DIR"
else
   rm -r "$ENV_DIR"
fi;
