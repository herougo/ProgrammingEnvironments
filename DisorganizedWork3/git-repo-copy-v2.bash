#!/bin/bash

printgreen() {
    printf "\e[0;32m$1\n\e[0m"
}
printyellow() {
    printf "\e[0;33m$1\n\e[0m"
}
printred() {
    printf "\e[0;31m$1\n\e[0m"
}
assertisnumber() {
    RE='^[0-9]+$'
    if ! [[ $1 =~ $RE ]] ; then
        echo "error: ${1} is not a number" >&2
        exit 1
    fi
}

# arguments
STOP_POINT=$1
START_POINT=0
NEW_BRANCH="$2"

# argument validation
OLD_BRANCH="`git rev-parse --abbrev-ref HEAD`"
assertisnumber $STOP_POINT
assertisnumber $START_POINT
if [[ "$NEW_BRANCH" == "" ]] ; then
    echo "error: you must specify a 2nd argument (branch name)" >&2
    exit 1
fi
if ! [[ $STOP_POINT > $START_POINT ]] ; then
    echo "error: 1st argument must be bigger than 0" >&2
    exit 1
fi
if [ "`git show-ref refs/heads/$NEW_BRANCH`" == "" ]; then
    echo "error: $NEW_BRANCH is not a valid branch. Exiting."
    exit 1
fi

# other error handling
if [[ "`git diff`" != "" ]] ; then
    echo "error: you have uncommitted changes (exiting)" >&2
    exit 1
fi
if [ "${OLD_BRANCH}" == "$NEW_BRANCH" ]; then
    echo "error: Current branch must be different from the 2nd argument. Exiting immediately"
    exit 1
fi
if [ "${OLD_BRANCH}" == "master" ]; then
    echo "error: Current branch is master. Exiting immediately"
    exit 1
fi

git checkout "$NEW_BRANCH"


STOP_MINUS_START=$((STOP_POINT-START_POINT))
STOP_POINT_PLUS_MORE=$((STOP_POINT+2))
assertisnumber $STOP_MINUS_START
assertisnumber $STOP_POINT_PLUS_MORE

LATEST_MESSAGES="`git show -s --format=%s -${STOP_POINT_PLUS_MORE}`"
# TO_IGNORE_LATEST=`echo "${LATEST_MESSAGES}" | head -${START_POINT}`
TO_COPY=`echo "${LATEST_MESSAGES}"| head -${STOP_POINT} | tail -${STOP_MINUS_START}`
TO_IGNORE_EARLIEST=`echo "${LATEST_MESSAGES}" | tail -2`

echo ""
# echo "Git commits to ignore (latest):"
# printred "${TO_IGNORE_LATEST}"
echo "Git commits to copy files from:"
printyellow "${TO_COPY}"
echo "Git commits to ignore (earliest):"
printred "${TO_IGNORE_EARLIEST}"

FILES_TO_CHECKOUT="`git diff --name-only HEAD~$STOP_POINT HEAD~$START_POINT`"

echo ""
echo "Files to checkout in $OLD_BRANCH"
printyellow "${FILES_TO_CHECKOUT}"
echo ""

git checkout "$OLD_BRANCH"

read -p "Proceed with repo file copy? (y/n)" _yesno
if [ "$_yesno" != "y" ] ; then
    echo "Did nothing: ${_yesno}"
    exit 0
fi


echo "$FILES_TO_CHECKOUT" | while read line ; do
    # https://stackoverflow.com/questions/2364147/how-to-get-just-one-file-from-another-branch
    printyellow "git checkout \"$NEW_BRANCH\" -- \"$line\""
    # this also works with paths with spaces
    git checkout "$NEW_BRANCH" -- "$line"
done

echo ""
printgreen "Done!"
echo "Note: these checkout changes won't show up on git diff, but if you git commit, they will be committed."
