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

I added tags and the time went up from 188 to 3096 secconds. which is 3096/60 = 51 minutes. 

I LEFT A SPACE IN THE TALLIES. running it again.

Galina is asking about the si sp cards for mcnp.

WHEW, DONE!!!!!

I got the effective map for deposition to detector, I still havent found an effective way to map this to the detectors tally 8 readings, but it at least shows the where the particles are coming from.

Ive discovered the fmesh tally, which might fix my issues with the cell based methods, but I want to commit to the data I have now. but I will try it out when we need higher resolution data.

Now I need to see if there is a significant difference between effective maps.

My current goals for this step are to generate all the base specs at 10x10x10 (should take 17 hours) and then to finish visualizing the effective maps.

Ill stop measuring the flux. 

Later I might also measure the deposition of the air heat by the detector, It might be useful later.

Ok, I have the sims running, ill make the visuals while they run on the test data.

I should finish the visualizations of the effective maps, then ill put them into a presentation.


ok, I just want to include the 50% barrier, the source cone and have a video cycle through the maps

Ive got all the presentations rendered, i just need the last two spectrums, and to put my the output data into excel sheets so they can read it.

Ive got it all running, i just need to update the excel sheet in the morning, and i can have all the emails in queve

presentation went great, i have a couple of prespectrives. I need to download the video.

Galina wants a 90% - 95% - 99% barrier, she also wants the detector and source included, so I will do that.