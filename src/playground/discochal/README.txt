Discochal script set
--------------------

This set of simple bash scripts is intended for the www.hacker.org homepage and might be useful if:
- you want to find out which challenges were solved by your competitors, but not by yourself
- you want to know who solved a particular challenge out of your watchlist
- you want to know if hints in the forum are given by someone who solved the challenge or not

It works in Cygwin under Windows and should work on any Linux flavour with bash (tested on Fedora 16).

You should first make all scripts executable with 
chmod 755 *.sh
then update the "database" with ./updatechals.sh . This initial step can take up to 5 hours and is optional if you can live with the dataset shipped with this package (updated 26.4.2012)! Finally create a user file with your username with ./discochal.sh yourusername

The list of commands you can use is:

./updatechals.sh
Updates the solver lists for each challenge by downloading them from hacker.org, it can take up to 5 hours to do this. This package already contains data updated on 26.4.2012, so you might perform this step only if you downloaded this package a long tiem after this date.

./updatesinglechal.sh 303
Updates the solvers of a single challenge id.

./discochal.sh Tilka
Creates a complete list of challenges solved by user Tilka. It takes about one minute to execute. After you launched this command, you can use diffchall.sh with the user Tilka  and the user will appear in whosolved.sh as well, as it will automatically be added to watched.cfg.

./diffchall.sh Tron dangermouse
Retrieves all challenges solved by Tron but not by dangermouse

./diffchall.sh dangermouse Tron
Retrieves all challenges solved by dangermouse but not by Tron (none to date :-)

./whosolved.sh "Anybody Out There" 
or
./whosolved.sh Anybody 
Retrieves all users who solved the challenge "Anybody Out There" which are listed in watched.cfg

./watched.sh
Shows all users you can currently analyze

./solvedby.sh dangermouse
Returns challenges solved by the user dangermouse

./removeuser.sh Tilka
Removes a user from watched.cfg previously added with discochal.sh

./updateusers.sh
Updates all users. It is good to launch this script when you already have users in your watchlist, after you did an update of the solvers with ./updatechals.sh

./cleanup.sh
Cleans up the entire database and the users. You then need to rerun updatechals.sh

(c) 2012 by dangermouse
Read my blog at http://www.advogato.org/person/dangermaus/
