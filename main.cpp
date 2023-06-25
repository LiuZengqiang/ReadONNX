// #include "attributeGetter.h"
#include "onnx.pb.h"
#include <cassert>
#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <string>
#include <vector>

/**
 * @brief 打印 dim
 * @param dim
 */
void print_dim(const ::onnx::TensorShapeProto_Dimension &dim)
{
    switch (dim.value_case())
    {
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << dim.dim_param();
        break;
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << dim.dim_value();
        break;
    default:
        assert(false && "should never happen");
    }
}
/**
 * @brief 打印 ValueInfoProto
 * @param info
 */
void print_io_info(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info)
{
    for (auto input_data : info)
    {
        auto shape = input_data.type().tensor_type().shape();
        std::cout << "  " << input_data.name() << ":";
        std::cout << "[";
        if (shape.dim_size() != 0)
        {
            int size = shape.dim_size();
            for (int i = 0; i < size - 1; ++i)
            {
                print_dim(shape.dim(i));
                std::cout << ", ";
            }
            print_dim(shape.dim(size - 1));
        }
        std::cout << "]\n";
    }
}

/**
 * @brief 打印 node.inputs，包括 name, data_type, dims, elements value
 * @param node
 * @param graph
 * @param graph_initializer_map
 */
void print_Inputs(const ::onnx::NodeProto &node, const ::onnx::GraphProto &graph,
                  std::map<std::string, int> &graph_initializer_map)
{

    std::cout << "........ inputs ........\n";

    for (auto input : node.input())
    {
        if (graph_initializer_map.find(input) != graph_initializer_map.end())
        {
            int idx = graph_initializer_map[input];

            auto tensor = graph.initializer()[idx];

            auto data_type = tensor.data_type();
            void *data_ptr = (void *)tensor.raw_data().data();
            size_t data_nbytes = tensor.raw_data().size();
            int data_size = 1;

            std::cout << "\tname:" << input << "\n";
            std::cout << "\tdata_type:" << ::onnx::TensorProto_DataType_Name(data_type) << "\n";
            std::cout << "\tdim: [";
            for (int i = 0; i < tensor.dims().size(); i++)
            {
                std::cout << tensor.dims()[i];

                data_size *= tensor.dims()[i];

                if (i != tensor.dims().size() - 1)
                {
                    std::cout << ",";
                }
                else
                {
                    std::cout << "]\n";
                }
            }

            std::cout << "\tdata(" << data_size << "): [";
            switch (data_type)
            {
            case ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
                float *temp_data_ptr = (float *)data_ptr;
                if (data_size > 10)
                {
                    for (int i = 0; i < 9; i++)
                    {
                        std::cout << temp_data_ptr[i] << ",";
                    }
                    std::cout << temp_data_ptr[9] << ",...]\n";
                }
                else
                {
                    for (int i = 0; i < data_size; i++)
                    {
                        std::cout << temp_data_ptr[i];
                        if (i != data_size - 1)
                        {
                            std::cout << ",";
                        }
                        else
                        {
                            std::cout << "]\n";
                        }
                    }
                }
            }
            break;
            case ::onnx::TensorProto_DataType::TensorProto_DataType_INT64: {
                /* code */
                long long *temp_data_ptr = (long long *)data_ptr;
                if (data_size > 10)
                {

                    for (int i = 0; i < 9; i++)
                    {
                        std::cout << temp_data_ptr[i] << ",";
                    }
                    std::cout << temp_data_ptr[9] << ",...]\n";
                }
                else
                {
                    for (int i = 0; i < data_size; i++)
                    {
                        std::cout << temp_data_ptr[i];
                        if (i != data_size - 1)
                        {
                            std::cout << ",";
                        }
                        else
                        {
                            std::cout << "]\n";
                        }
                    }
                }
            }
            break;
            default:
                break;
            }
        }
        else
        {
            std::cout << "\tname:" << input << "\n";
        }
    }

    std::cout << "........................\n";
}

/**
 * @brief 打印 node.output
 * @param node
 */
void print_Outputs(const ::onnx::NodeProto &node)
{

    std::cout << "........ ouputs ........\n";
    for (auto output : node.output())
    {
        std::cout << "\tname:" << output << "\n";
    }
    std::cout << "........................\n";
}

/**
 * @brief 打印 node.atrributes
 * @param node
 */
void print_Attribute(const ::onnx::NodeProto &node)
{
    std::cout << "........ attribute ........\n";

    for (int i = 0; i < node.attribute_size(); i++)
    {
        const ::onnx::AttributeProto &attri = node.attribute()[i];

        std::cout << "\tname:" << attri.name() << "\n";
        std::cout << "\ttype:" << ::onnx::AttributeProto_AttributeType_Name(attri.type()) << "\n";
        std::cout << "\tdata: [";

        if (attri.type() == ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT)
        {
            std::cout << attri.f() << "]\n";
        }
        else if (attri.type() == ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS)
        {
            if (attri.floats_size() > 10)
            {
                for (int j = 0; j < 9; j++)
                {
                    std::cout << attri.floats()[j] << ",";
                }
                std::cout << attri.floats()[10] << ",...]\n";
            }
            else
            {
                for (int j = 0; j < attri.floats_size(); j++)
                {
                    std::cout << attri.floats()[j];
                    if (j == attri.floats_size() - 1)
                    {
                        std::cout << "]\n";
                    }
                    else
                    {
                        std::cout << ",";
                    }
                }
            }
        }
        else if (attri.type() == ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT)
        {
            std::cout << attri.i() << "]\n";
        }
        else if (attri.type() == ::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS)
        {
            if (attri.ints_size() > 10)
            {
                for (int j = 0; j < 9; j++)
                {
                    std::cout << attri.ints()[j] << ",";
                }
                std::cout << attri.ints()[10] << ",...]\n";
            }
            else
            {
                for (int j = 0; j < attri.ints_size(); j++)
                {
                    std::cout << attri.ints()[j];
                    if (j == attri.ints_size() - 1)
                    {
                        std::cout << "]\n";
                    }
                    else
                    {
                        std::cout << ",";
                    }
                }
            }
        }
        else
        {
            std::cout << "Unsupport data type:" << ::onnx::AttributeProto_AttributeType_Name(attri.type()) << std::endl;
        }
    }
    std::cout << "...........................\n";
}

/**
 * @brief 打印 node
 * @param node
 * @param graph
 * @param graph_initializer_map
 */
void print_Node(const ::onnx::NodeProto &node, const ::onnx::GraphProto &graph,
                std::map<std::string, int> &graph_initializer_map)
{
    std::cout << "======== Node ========\n";
    std::cout << "name:" << node.name() << "\n";
    std::cout << "op_type:" << node.op_type() << "\n";

    print_Attribute(node);
    print_Inputs(node, graph, graph_initializer_map);
    print_Outputs(node);
    std::cout << "======================\n";
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        std::cout << "请指定模型！"
                  << "\n";
        return -1;
    }
    std::cout << "Usage: " << argv[1] << "\n";

    /* 读取模型文件.onnx */
    std::ifstream input(argv[1], std::ios::ate | std::ios::binary); // open file and move current position
    // in file to the end

    std::streamsize size = input.tellg(); // get current position in file
    input.seekg(0, std::ios::beg);        // move to start of file

    std::vector<char> buffer(size);
    input.read(buffer.data(), size); // read raw data 并且将数据存在 buffer 中

    google::protobuf::io::CodedInputStream input_stream(reinterpret_cast<const uint8_t *>(buffer.data()), size);
    input_stream.SetTotalBytesLimit(size + 1, 0);

    onnx::ModelProto model;

    model.ParseFromCodedStream(&input_stream);

    auto graph = model.graph();

    /* 建立 initializer:index 映射 */

    std::map<std::string, int> graph_initializer_map;
    for (int i = 0; i < graph.initializer().size(); i++)
    {
        graph_initializer_map.insert({graph.initializer()[i].name(), i});
    }
    /* 打印 graph inputs 信息 */
    std::cout << "graph inputs:\n";
    print_io_info(graph.input());

    /* 打印 graph 各 node 的信息 */
    for (int i = 0; i < graph.node().size(); i++)
    {
        print_Node(graph.node()[i], graph, graph_initializer_map);
    }
    /* 打印 graph outputs 信息 */
    std::cout << "graph outputs:\n";
    print_io_info(graph.output());
    return 0;
}