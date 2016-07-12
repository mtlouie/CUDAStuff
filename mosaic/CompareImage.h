void CompareImage(ImageData srcImage, ImageData ResampledTest, int i, int *index_array, long *min_rms_array, /* int * orientation_array, */
		  int nx_dim, int ny_dim);

void ReplaceInImage(int img, int *index_array,  /* orientation_array, */ 
		    int nx_dim, int ny_dim, 
		    ImageData FinalImage, ImageData ResampledTest);

