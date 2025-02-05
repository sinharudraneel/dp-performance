void clamp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
    // Get operands
    const operand_info &dst = pI->dst();     // destination
    const operand_info &src1 = pI->src1();   // value to clamp
    const operand_info &src2 = pI->src2();   // min value
    const operand_info &src3 = pI->src3();   // max value

    // Get values from operands
    unsigned i_type = pI->get_type();
    ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
    ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
    ptx_reg_t src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);
    ptx_reg_t data;

    // Perform clamping based on data type
    switch (i_type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE: {
            // Signed integer clamping
            data.s64 = src1_data.s64;
            if (data.s64 < src2_data.s64) data.s64 = src2_data.s64;
            if (data.s64 > src3_data.s64) data.s64 = src3_data.s64;
            break;
        }
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE: {
            // Unsigned integer clamping
            data.u64 = src1_data.u64;
            if (data.u64 < src2_data.u64) data.u64 = src2_data.u64;
            if (data.u64 > src3_data.u64) data.u64 = src3_data.u64;
            break;
        }
        case F16_TYPE: {
            // Half-precision float clamping
            data.f16 = src1_data.f16;
            if (data.f16 < src2_data.f16) data.f16 = src2_data.f16;
            if (data.f16 > src3_data.f16) data.f16 = src3_data.f16;
            break;
        }
        case F32_TYPE: {
            // Single-precision float clamping
            data.f32 = src1_data.f32;
            if (isnan(src1_data.f32)) {
                // Handle NaN case based on your requirements
                data.f32 = src2_data.f32;
            } else {
                if (data.f32 < src2_data.f32) data.f32 = src2_data.f32;
                if (data.f32 > src3_data.f32) data.f32 = src3_data.f32;
            }
            break;
        }
        case F64_TYPE:
        case FF64_TYPE: {
            // Double-precision float clamping
            data.f64 = src1_data.f64;
            if (std::isnan(src1_data.f64)) {
                data.f64 = src2_data.f64;
            } else {
                if (data.f64 < src2_data.f64) data.f64 = src2_data.f64;
                if (data.f64 > src3_data.f64) data.f64 = src3_data.f64;
            }
            break;
        }
        default:
            printf("Execution error: type mismatch with instruction\n");
            assert(0);
            break;
    }

    // Store result
    thread->set_operand_value(dst, data, i_type, thread, pI);
}
