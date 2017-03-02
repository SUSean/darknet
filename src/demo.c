#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#ifdef MPI
#include <mpi.h>
#define frameTag 1
#define predictTag 2
#define timeTag 3
#endif

#define FRAMES 4
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[100];
static image origin_images[100];
static float *avg;
static float nms = .4;

void *fetch_in_thread(void *ptr)
{
	in = get_image_from_stream(cap);
	if(!in.data){
		error("Stream closed.");
	}
	double resizeTime;
	resizeTime = MPI_Wtime();
	in_s = resize_image(in, net.w, net.h);
	printf("Resize image in %lf seconds\n",MPI_Wtime()-resizeTime);
	return 0;
}
void fetch_frames()
{
	for(int i = 0; i < 100; i++){
		in = get_image_from_stream(cap);
		if(!in.data){
			error("Stream closed.");
		}
		double resizeTime;
		resizeTime = MPI_Wtime();
		in_s = resize_image(in, net.w, net.h);
		printf("Resize image in %lf seconds\n",MPI_Wtime()-resizeTime);
		origin_images[i]=in;
		images[i]=in_s;
	}
}
#ifdef MPI
void detect_frame(layer l,int index, int frame)
{
	//mean_arrays(predictions, FRAMES, l.outputs, avg);
	l.output = predictions[index];
	if(l.type == DETECTION){
		get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
	} else if (l.type == REGION){
		get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0, demo_hier_thresh);
	} else {
		error("Last layer must produce detections\n");
	}
	if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	//printf("\033[2J");
	//printf("\033[1;1H");
	printf("\nFPS:%.1f\n",fps);
	printf("Objects:\n\n");

	det = origin_images[frame];

	draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
}
#else
void *detect_in_thread(void *ptr)
{
	layer l = net.layers[net.n-1];
	float *X = det_s.data;
	float time = clock();
	float *prediction = network_predict(net, X);
	printf("Prediction in %f seconds.\n", sec(clock()-time));
	memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
	mean_arrays(predictions, FRAMES, l.outputs, avg);
	l.output = avg;

	free_image(det_s);
	if(l.type == DETECTION){
		get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
	} else if (l.type == REGION){
		get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0, demo_hier_thresh);
	} else {
		error("Last layer must produce detections\n");
	}
	if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nFPS:%.1f\n",fps);
	printf("Objects:\n\n");

	images[demo_index] = det;
	det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
	demo_index = (demo_index + 1)%FRAMES;

	draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

	return 0;
}
#endif
double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time,NULL)){
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
	//skip = frame_skip;
#ifdef MPI
	int rank,size;
	MPI_Status status;
	MPI_Request req;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	printf("Hello world from node %s, rank %d out of %d processes\n",processor_name, rank, size);
#endif
	image **alphabet = load_alphabet();
	int delay = frame_skip;
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	demo_hier_thresh = hier_thresh;
	printf("Demo\n");
	net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);

	srand(2222222);

	layer l = net.layers[net.n-1];
	int j;

	avg = (float *) calloc(l.outputs, sizeof(float));
	for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
	for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

	boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));

#ifdef MPI
	if(rank == 0){
		if(filename){
			printf("video file: %s\n", filename);
			cap = cvCaptureFromFile(filename);
		}else{
			cap = cvCaptureFromCAM(cam_index);
		}
		if(!cap) error("Couldn't connect to webcam.\n");

		if(!prefix){
			cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
			cvMoveWindow("Demo", 0, 0);
			cvResizeWindow("Demo", 1352, 1013);
		}
	}
	int time = -1;
	int now = 0;
	int temp;
	float *prediction = (float *) malloc(l.outputs* sizeof(float));
	double start,end;
	//pthread_t fetch_thread;
	
	if(rank == 0){
		double totalstart = MPI_Wtime(); 
		double before = MPI_Wtime();
		double after,curr;
		int returnRank;
		//if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
		fetch_frames();
		while(1){
			time ++;
			
			while(time < size-1){
				printf("Frame = %d\n",time);
				//pthread_join(fetch_thread, 0);
				//images[time - 1] = in;
				//det_s = in_s;
				//if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
				MPI_Send(images[time].data,net.w*net.h*3,MPI_FLOAT,time+1,frameTag,MPI_COMM_WORLD);
				MPI_Send(&time,1,MPI_INT,time+1,timeTag,MPI_COMM_WORLD);
				printf("Send frame to %d\n",time);
				time ++;
				//free_image(det_s);
			}
			
			start = MPI_Wtime();
			MPI_Recv(prediction,l.outputs,MPI_FLOAT,MPI_ANY_SOURCE,predictTag,MPI_COMM_WORLD,&status);
			returnRank=status.MPI_SOURCE;
			MPI_Recv(&temp,1,MPI_INT,returnRank,timeTag,MPI_COMM_WORLD,&status);
			end = MPI_Wtime();
			printf("Receive prediction from %d in %lf seconds.\n", returnRank, end - start);

			if(temp > now){
				now = temp;
				start = MPI_Wtime();
				memcpy(predictions[returnRank-1], prediction, l.outputs*sizeof(float));
				detect_frame(l,returnRank-1,temp);
				disp = det;
				end = MPI_Wtime();
				printf("Detect in %lf seconds.\n",end - start);
				start = MPI_Wtime();
				if(!prefix){
   	            	//show_image(disp, "Demo");
                	int c = cvWaitKey(1);
                   	if (c == 10){
                       	if(frame_skip == 0) frame_skip = 60;
                       	else if(frame_skip == 4) frame_skip = 0;
                       	else if(frame_skip == 60) frame_skip = 4;  
						else frame_skip = 0;
                	}
                }else{
                    char buff[256];
                    sprintf(buff, "%s_%08d", prefix, time);
                   	save_image(disp, buff);
               	}
               	if(delay == 0){
					free_image(disp);
                }
				--delay;
               	if(delay < 0){
              		delay = frame_skip;
              	}
				end = MPI_Wtime();
				printf("Display in %lf seconds.\n",end - start);
			}
			after = MPI_Wtime();
			printf("Total time = %lf\n\n\n",after - before);
			curr = 1./(after - before);
			fps = curr;
			before = after;
			
			//printf("prediction is at %p\n", &prediction);
			//printf("det_s is at %p\n", &det_s);
			//printf("image%d is at %p\n", returnRank, &images[returnRank - 1]);
			//images[returnRank - 1] = make_image(1,1,3);
			
			start = MPI_Wtime();
			//pthread_join(fetch_thread, 0);
			//images[returnRank - 1] = in;
			//det_s = in_s;
			//if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
			end = MPI_Wtime();
			printf("Fetch time %lf seconds.\n",end - start);
			printf("Frame = %d\n",time);
			
			start = MPI_Wtime();
			MPI_Send(images[time].data,net.w*net.h*3,MPI_FLOAT,returnRank,frameTag,MPI_COMM_WORLD);
			MPI_Send(&time,1,MPI_INT,returnRank,timeTag,MPI_COMM_WORLD);
			end = MPI_Wtime();
			//free_image(det_s);
			printf("Send Frame to %d in %lf seconds.\n", returnRank, end - start);
		}
		printf("Average fps %lf.\n",(double) time / (MPI_Wtime()-totalstart));
	}
	else{
		float *X  = (float *) malloc(net.w*net.h*3* sizeof(float));
		while(1){
			MPI_Recv(X,net.w*net.h*3,MPI_FLOAT,0,frameTag,MPI_COMM_WORLD,&status);
			MPI_Recv(&time,1,MPI_INT,0,timeTag,MPI_COMM_WORLD,&status);
			printf("%d Receive frame\n",rank);
			start = MPI_Wtime();
			prediction = network_predict(net, X);
			end = MPI_Wtime();
			printf("%d Prediction in %lf seconds.\n", rank, end - start);
			MPI_Send(prediction,l.outputs,MPI_FLOAT,0,predictTag,MPI_COMM_WORLD);
			MPI_Send(&time,1,MPI_INT,0,timeTag,MPI_COMM_WORLD);
		}
		free(X);
	}
	free(prediction);

#else
	if(filename){
		printf("video file: %s\n", filename);
		cap = cvCaptureFromFile(filename);
	}else{
		cap = cvCaptureFromCAM(cam_index);
	}

	if(!cap) error("Couldn't connect to webcam.\n");

	pthread_t fetch_thread;
	pthread_t detect_thread;

	fetch_in_thread(0);
	det = in;
	det_s = in_s;

	fetch_in_thread(0);
	detect_in_thread(0);
	disp = det;
	det = in;
	det_s = in_s;

	for(j = 0; j < FRAMES/2; ++j){
		fetch_in_thread(0);
		detect_in_thread(0);
		disp = det;
		det = in;
		det_s = in_s;
	}

	int count = 0;
	if(!prefix){
		cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
		cvMoveWindow("Demo", 0, 0);
		cvResizeWindow("Demo", 1352, 1013);
	}

	double before = get_wall_time();

	while(1){
		++count;
    		printf("frame = %d\n",count);
		if(1){
			if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
			if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

			if(!prefix){
				show_image(disp, "Demo");
				int c = cvWaitKey(1);
				if (c == 10){
					if(frame_skip == 0) frame_skip = 60;
					else if(frame_skip == 4) frame_skip = 0;
					else if(frame_skip == 60) frame_skip = 4;   
					else frame_skip = 0;
				}
			}else{
				char buff[256];
				sprintf(buff, "%s_%08d", prefix, count);
				save_image(disp, buff);
			}

			pthread_join(fetch_thread, 0);
			pthread_join(detect_thread, 0);

			if(delay == 0){
				free_image(disp);
				disp  = det;
			}
			det   = in;
			det_s = in_s;
		}else {
			fetch_in_thread(0);
			det   = in;
			det_s = in_s;
			detect_in_thread(0);
			if(delay == 0) {
				free_image(disp);
				disp = det;
			}
			show_image(disp, "Demo");
			cvWaitKey(1);
		}
		--delay;
		if(delay < 0){
			delay = frame_skip;

			double after = get_wall_time();
			float curr = 1./(after - before);
			fps = curr;
			before = after;
		}
	}
#endif
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
	fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

