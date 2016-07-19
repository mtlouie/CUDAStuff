void CompareImage(ImageData srcImage, ImageData ResampledTest, int i, int *index_array, long *min_rms_array, 
		  int nx_dim, int ny_dim, int do_grayscale); /* int * orientation_array, */

void ReplaceInImage(int img, int *index_array,  
		    int nx_dim, int ny_dim, 
		    ImageData FinalImage, ImageData ResampledTest, int do_grayscale);/* orientation_array, */ 

