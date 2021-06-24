#include "read_mnist.h"

#include "neural_net.h"

inline bool is_little_endian() {
    // gets stored as 00000000 00000001 <- big endian
    // or 10000000 00000000 <- little endian
    uint16_t num = 1;
    // char* points to first byte
    return *reinterpret_cast<char*>(&num) == 1;
}

// swap order of bytes
inline int32_t swap(uint32_t num) {
    int32_t result = num & 0xff;
    result         = (result << 8) | ((num >> 8) & 0xff);
    result         = (result << 8) | ((num >> 16) & 0xff);
    result         = (result << 8) | ((num >> 24) & 0xff);
    return result;
}

inline int32_t get_int32_t(std::ifstream& file) {
    int32_t num = 0;
    file.read(reinterpret_cast<char*>(&num), 4);
    // number is in big endian
    if(is_little_endian())
        num = swap(num);
    return num;
}

NeuralNet::Data load_data(const std::string& images_path, const std::string& labels_path) {
    try {
        ////////////
        // images //
        ////////////
        std::ifstream images_file;
        images_file.open(images_path, std::ios::in | std::ios::binary);
        if(!images_file.is_open())
            raise_critical("Can't open images file: {}", images_path);

        // magic number
        if(get_int32_t(images_file) != 2051)
            raise_critical("images file '{}' has an unsupported magic number", images_path);
        // data set dimensions
        int32_t images_amount = get_int32_t(images_file);
        int32_t n_rows        = get_int32_t(images_file);
        int32_t n_cols        = get_int32_t(images_file);

        ////////////
        // labels //
        ////////////
        std::ifstream labels_file;
        labels_file.open(labels_path, std::ios::in | std::ios::binary);
        if(!images_file.is_open())
            raise_critical("Can't open labels file: {}", images_path);

        // magic number
        if(get_int32_t(labels_file) != 2049)
            raise_critical("labels file '{}' has an unsupported magic number", labels_path);
        // data set dimensions
        int32_t labels_amount = get_int32_t(labels_file);
        if(images_amount != labels_amount)
            raise_critical("images file '{}' and labels file '{}' don't have same amount ({} and {}) of data sets",
                           images_path, labels_path, images_amount, labels_amount);

        //////////////////
        // read dataset //
        //////////////////
        // one column per data set
        NeuralNet::Data data(images_amount, n_rows * n_cols, 10);
        for(int i = 0; i < images_amount; ++i) {
            for(int pixel_idx = 0; pixel_idx < n_rows * n_cols; ++pixel_idx) {
                uint8_t pixel;
                images_file.read(reinterpret_cast<char*>(&pixel), 1);
                data.get_x().at(pixel_idx, i) = pixel / 255.0f;
            }
            uint8_t label;
            labels_file.read(reinterpret_cast<char*>(&label), 1);
            data.get_y().at(label, i) = 1.0f;
        }

        images_file.close();
        labels_file.close();
        // todo: bad copy?
        return data;
    } catch(const std::exception& ex) {
        raise_critical("{} error parsing MNIST file '{}' or '{}'", ex.what(), images_path, labels_path);
    }
}
