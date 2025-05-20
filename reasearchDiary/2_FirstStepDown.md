Im gonna start out by testing a the coconut shells at the three resolutions. I need to make sure the code reads for any size effective map.

It works, at 1e4  1 cell takes 2 seconds to run. 8 cells takes about the same, but 100 cells take 31 seconds. 


Im realizing, giving the cells their own tallies it could slow it down unnecessarily. something to keep note of for the next time.

I am worried combining all the tallies wont work but i can atleast consolidate the soil tallies.

I did this completely only to find out that the fortran program cannot handle that many f bins, so instead i have to switch back to doing a bunch of tallies

it cant handle alot of things. I cut the time bins to make it fit in a 10^3. Ill ask the fourm if you can increase the memory size or something.

I might have to go back later and try to do atlease INS bin.

Maybe in the future Ill try to do chunk soil tallies, where I do a few soil cells at a time. I know it works with 8 atleast


IT LOOKS GREAT! I realize I might need to specify the intensity measure to be neutrons and not photons.

may 20 / 25

todo
- Neutrons instead of photons
- deposition to detector map
- flux maps

once i finish this i can leave the spectrums running at 1e9 for high res results

I start by fixing the scripts. You know what, ill add photons AND neutrons to the tallies. need to save that memory. Might kill tally 18 and 28. Could be doing GEB in post

OMGGGG, the flags's are not seperated into seperate bins!?
im gonna try to use ft tag on the tallies, also ill erase the geb card and the associated tally for now, that can definetly be post processed