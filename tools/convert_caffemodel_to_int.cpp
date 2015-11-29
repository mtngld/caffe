#include <stdio.h>  // for snprintf
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;
using namespace caffe;

void make_blobproto_int (BlobProto* proto)
{
  float max_int = 127;
  float scale  = std::max(std::abs(proto->mean() - proto->max()),std::abs(proto->mean() - proto->min()));
  float shift  = proto->mean();
  for (int i = 0; i < proto->data_size(); ++i) {
    proto->add_int_data(int(max_int*(proto->data(i)-shift)/scale));
  }
  proto->clear_data();
}

void make_layer_int(LayerParameter* param) {
    LOG(ERROR) << "(float->int)Working on " << param->name();
    for (int i = 0; i < param->blobs_size(); ++i) {
      make_blobproto_int(param->mutable_blobs(i));
  }
}

void make_net_int(NetParameter* param) {
    for (int i = 0; i < param->layer_size(); ++i) {
      make_layer_int(param->mutable_layer(i));
    }
}

void make_blobproto_float(BlobProto* proto)
{
  float max_int = 127;
  float scale  = std::max(std::abs(proto->mean() - proto->max()),std::abs(proto->mean() - proto->min()));
  float shift  = proto->mean();
  LOG(ERROR) << "mean = " << proto->mean() << ", max = " << proto->max() << ", min = " << proto->min() << ", scale = " << scale << ", shift = " << shift;
  LOG(ERROR) << "abs(proto->mean() - proto->max()) = " << std::abs(proto->mean() - proto->max()) << ", abs(proto->mean() - proto->min())" << std::abs(proto->mean() - proto->min());

  for (int i = 0; i < proto->int_data_size(); ++i) {
      proto->add_data((scale/max_int)*float(proto->int_data(i))+shift);
      //LOG(ERROR) << "(int->float) "  << proto->int_data(i) << "->"<< proto->data(i);
  }
  proto->clear_int_data();
}

void make_layer_float(LayerParameter* param) {
    LOG(ERROR) << "(int->float) Working on " << param->name();
    for (int i = 0; i < param->blobs_size(); ++i) {
      make_blobproto_float(param->mutable_blobs(i));
  }
}

void make_net_float(NetParameter* param) {
    for (int i = 0; i < param->layer_size(); ++i) {
      make_layer_float(param->mutable_layer(i));
    }
}



template<typename Dtype>
int convert_caffe_model_to_int_pipeline(int argc, char** argv)
{
    const int num_required_args = 3;
    if (argc < num_required_args)
    {
        LOG(ERROR) <<
                      "\nconvert_caffe_model_to_int -- Nov 2015\n"
                      "Edited by: Matan Goldman\n"
                      "This program loads a pretrained network and converts the weights to int\n"
                      "In order to save space\n"
                      "\tUsage: \n"
                      "\t<path_to_weight_file> <path_to_model_prototxt> <path_to_output_weight_file>";
        return 1;
    }



    std::string pretrained_binary(argv[1]); // path to pretrained binary file
    std::string feature_extraction_proto(argv[2]); // path to feature extraction proto file
    LOG(ERROR) << "Original binary: " << pretrained_binary;
    LOG(ERROR) << "Original proto:  "  << feature_extraction_proto;

    shared_ptr<Net<Dtype> > original_net(new Net<Dtype>(feature_extraction_proto, caffe::TEST));

    caffe::NetParameter net_param;
    original_net->CopyTrainedLayersFrom(pretrained_binary);   /// load pretrained weights
    original_net->ToProto(&net_param, false);

    //Compress
    make_net_int(&net_param);
    WriteProtoToBinaryFile(net_param, pretrained_binary + ".compressed");

    //Restore
    make_net_float(&net_param);
    WriteProtoToBinaryFile(net_param, pretrained_binary + ".restored");


    LOG(ERROR)<< "Successfully extracted the features!";
    return 0;
}


int main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);
  return convert_caffe_model_to_int_pipeline<float>(argc, argv);
}
