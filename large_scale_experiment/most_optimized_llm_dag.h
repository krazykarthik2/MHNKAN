/* Automatically compiled EML-KAN LLM C++ DAG equations */
#ifndef EML_KAN_LLM_DAG_H
#define EML_KAN_LLM_DAG_H

#include <math.h>

inline float softplus_stable(float z) {
    if (z > 20.0f) return z;
    if (z < -20.0f) return 0.0f;
    return logf(1.0f + expf(z));
}

// =================== BLOCK 0 FFN1 LAYER ===================
void evaluate_block_0_ffn1(const float* features, float* output_logits) {
    // Node 0
    float z_0 = 0.0f;
    z_0 += features[0] * 0.245548f;
    z_0 += features[1] * 0.145951f;
    z_0 += features[2] * 0.134056f;
    z_0 += features[3] * -0.186038f;
    z_0 += features[5] * 0.253597f;
    z_0 += features[11] * 0.122319f;
    z_0 += features[12] * 0.112086f;
    z_0 += features[14] * 0.132852f;
    z_0 += features[16] * -0.276367f;
    z_0 += features[17] * 0.143687f;
    z_0 += features[18] * -0.115135f;
    z_0 += features[19] * -0.111611f;
    z_0 += features[20] * -0.179698f;
    z_0 += features[21] * -0.164034f;
    z_0 += features[29] * -0.116920f;
    z_0 += features[31] * 0.119564f;
    z_0 += features[33] * -0.253456f;
    z_0 += features[34] * 0.264309f;
    z_0 += features[38] * -0.158483f;
    z_0 += features[39] * -0.176910f;
    z_0 += features[40] * -0.171088f;
    z_0 += features[42] * -0.179445f;
    z_0 += features[43] * 0.189061f;
    z_0 += features[45] * -0.117398f;
    z_0 += features[49] * -0.144532f;
    z_0 += features[54] * -0.259684f;
    z_0 += features[55] * -0.193738f;
    z_0 += features[61] * -0.199782f;
    z_0 += features[62] * -0.220016f;
    float out_0 = 0.243276f * z_0;
    {
        float arg_x = -0.098395f * z_0 + -0.083634f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.046304f * z_0 + 0.090146f) + 1e-6f;
        out_0 += 0.028480f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.110116f * z_0 + -0.085878f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.108787f * z_0 + 0.088826f) + 1e-6f;
        out_0 += 0.007532f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.038867f * z_0 + -0.033963f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.046231f * z_0 + 0.029356f) + 1e-6f;
        out_0 += 0.050246f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.077614f * z_0 + -0.088355f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.135317f * z_0 + 0.086751f) + 1e-6f;
        out_0 += 0.011006f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[0] = out_0;

    // Node 1
    float z_1 = 0.0f;
    z_1 += features[6] * 0.180569f;
    z_1 += features[8] * 0.188882f;
    z_1 += features[10] * -0.127093f;
    z_1 += features[11] * 0.151407f;
    z_1 += features[12] * 0.148950f;
    z_1 += features[14] * -0.122377f;
    z_1 += features[17] * 0.153030f;
    z_1 += features[18] * 0.160499f;
    z_1 += features[20] * -0.193494f;
    z_1 += features[22] * 0.122586f;
    z_1 += features[23] * 0.217960f;
    z_1 += features[25] * -0.150787f;
    z_1 += features[29] * 0.224257f;
    z_1 += features[31] * -0.202125f;
    z_1 += features[32] * 0.126161f;
    z_1 += features[35] * -0.201509f;
    z_1 += features[38] * -0.194646f;
    z_1 += features[40] * -0.238591f;
    z_1 += features[41] * 0.213738f;
    z_1 += features[47] * 0.117115f;
    z_1 += features[49] * -0.231549f;
    z_1 += features[50] * 0.122138f;
    z_1 += features[54] * 0.187184f;
    z_1 += features[56] * -0.223122f;
    z_1 += features[58] * -0.238157f;
    z_1 += features[59] * -0.128771f;
    z_1 += features[60] * 0.181924f;
    z_1 += features[61] * 0.162786f;
    z_1 += features[62] * 0.266356f;
    z_1 += features[63] * -0.177476f;
    float out_1 = 0.222567f * z_1;
    {
        float arg_x = 0.117641f * z_1 + -0.003438f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.121339f * z_1 + 0.014779f) + 1e-6f;
        out_1 += 0.056016f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.125789f * z_1 + -0.044115f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.101715f * z_1 + 0.053992f) + 1e-6f;
        out_1 += -0.006496f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.083248f * z_1 + -0.005025f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.100280f * z_1 + 0.005641f) + 1e-6f;
        out_1 += 0.050840f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.083749f * z_1 + -0.061055f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.128748f * z_1 + 0.059518f) + 1e-6f;
        out_1 += -0.000315f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[1] = out_1;

    // Node 2
    float z_2 = 0.0f;
    z_2 += features[0] * 0.136232f;
    z_2 += features[2] * -0.254973f;
    z_2 += features[3] * 0.207332f;
    z_2 += features[4] * -0.204625f;
    z_2 += features[6] * 0.274582f;
    z_2 += features[7] * 0.321865f;
    z_2 += features[10] * -0.200867f;
    z_2 += features[11] * -0.175885f;
    z_2 += features[12] * 0.214642f;
    z_2 += features[13] * 0.115979f;
    z_2 += features[14] * -0.173678f;
    z_2 += features[15] * -0.311862f;
    z_2 += features[16] * 0.376235f;
    z_2 += features[17] * -0.115486f;
    z_2 += features[19] * 0.311787f;
    z_2 += features[20] * -0.228211f;
    z_2 += features[22] * -0.171892f;
    z_2 += features[24] * 0.167208f;
    z_2 += features[27] * -0.262444f;
    z_2 += features[29] * 0.362558f;
    z_2 += features[30] * -0.212110f;
    z_2 += features[31] * 0.252749f;
    z_2 += features[32] * 0.160596f;
    z_2 += features[34] * -0.182261f;
    z_2 += features[38] * 0.143899f;
    z_2 += features[39] * 0.212567f;
    z_2 += features[41] * 0.178834f;
    z_2 += features[42] * 0.202313f;
    z_2 += features[44] * 0.405872f;
    z_2 += features[45] * -0.297680f;
    z_2 += features[46] * -0.220483f;
    z_2 += features[47] * -0.235376f;
    z_2 += features[48] * 0.114754f;
    z_2 += features[49] * 0.354187f;
    z_2 += features[50] * -0.295266f;
    z_2 += features[51] * -0.198155f;
    z_2 += features[52] * 0.188442f;
    z_2 += features[53] * -0.174841f;
    z_2 += features[54] * -0.123856f;
    z_2 += features[55] * -0.159940f;
    z_2 += features[57] * 0.253392f;
    z_2 += features[58] * -0.284994f;
    z_2 += features[59] * -0.161672f;
    z_2 += features[62] * 0.112432f;
    z_2 += features[63] * -0.189919f;
    float out_2 = 0.298203f * z_2;
    {
        float arg_x = -0.279034f * z_2 + -0.071158f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.190517f * z_2 + 0.103662f) + 1e-6f;
        out_2 += -0.037186f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.467995f * z_2 + 0.198991f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.342196f * z_2 + 0.004062f) + 1e-6f;
        out_2 += -0.156825f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.116901f * z_2 + -0.039921f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.177242f * z_2 + 0.044026f) + 1e-6f;
        out_2 += 0.054638f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.077494f * z_2 + -0.186166f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.093690f * z_2 + 0.181059f) + 1e-6f;
        out_2 += 0.000200f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[2] = out_2;

    // Node 3
    float z_3 = 0.0f;
    z_3 += features[1] * -0.154998f;
    z_3 += features[6] * -0.179840f;
    z_3 += features[7] * 0.208194f;
    z_3 += features[8] * -0.116054f;
    z_3 += features[9] * 0.177298f;
    z_3 += features[11] * -0.229004f;
    z_3 += features[12] * -0.160395f;
    z_3 += features[13] * -0.150591f;
    z_3 += features[14] * 0.235431f;
    z_3 += features[16] * 0.176867f;
    z_3 += features[19] * -0.205643f;
    z_3 += features[20] * 0.121640f;
    z_3 += features[22] * 0.138660f;
    z_3 += features[23] * -0.248023f;
    z_3 += features[24] * -0.111346f;
    z_3 += features[25] * 0.148706f;
    z_3 += features[26] * -0.282575f;
    z_3 += features[27] * 0.213109f;
    z_3 += features[31] * 0.157396f;
    z_3 += features[32] * -0.222035f;
    z_3 += features[34] * 0.181080f;
    z_3 += features[35] * 0.272882f;
    z_3 += features[38] * 0.156628f;
    z_3 += features[40] * 0.140992f;
    z_3 += features[41] * 0.141893f;
    z_3 += features[42] * -0.301047f;
    z_3 += features[47] * 0.116161f;
    z_3 += features[48] * -0.174998f;
    z_3 += features[49] * -0.261648f;
    z_3 += features[53] * 0.193906f;
    z_3 += features[54] * -0.203411f;
    z_3 += features[58] * 0.205222f;
    z_3 += features[59] * 0.192537f;
    z_3 += features[60] * 0.159620f;
    z_3 += features[61] * -0.235875f;
    z_3 += features[63] * 0.147109f;
    float out_3 = 0.224593f * z_3;
    {
        float arg_x = -0.110492f * z_3 + 0.063748f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.069920f * z_3 + -0.024707f) + 1e-6f;
        out_3 += -0.058266f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.159437f * z_3 + -0.038100f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.163688f * z_3 + 0.068426f) + 1e-6f;
        out_3 += 0.019138f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.169196f * z_3 + -0.015548f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.174377f * z_3 + 0.047715f) + 1e-6f;
        out_3 += 0.038576f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.087836f * z_3 + 0.050829f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.129567f * z_3 + -0.035289f) + 1e-6f;
        out_3 += -0.103766f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[3] = out_3;

    // Node 4
    float z_4 = 0.0f;
    z_4 += features[0] * 0.133752f;
    z_4 += features[1] * -0.228119f;
    z_4 += features[6] * 0.115491f;
    z_4 += features[7] * 0.152232f;
    z_4 += features[10] * 0.246737f;
    z_4 += features[11] * 0.200647f;
    z_4 += features[12] * 0.233116f;
    z_4 += features[14] * -0.191283f;
    z_4 += features[18] * 0.161867f;
    z_4 += features[20] * -0.266533f;
    z_4 += features[21] * 0.145933f;
    z_4 += features[22] * -0.168947f;
    z_4 += features[23] * 0.190140f;
    z_4 += features[24] * 0.141643f;
    z_4 += features[25] * -0.140403f;
    z_4 += features[27] * -0.178926f;
    z_4 += features[28] * 0.215139f;
    z_4 += features[31] * -0.254294f;
    z_4 += features[34] * -0.248097f;
    z_4 += features[35] * -0.125390f;
    z_4 += features[39] * -0.144355f;
    z_4 += features[40] * -0.210793f;
    z_4 += features[41] * 0.126221f;
    z_4 += features[42] * 0.134996f;
    z_4 += features[44] * 0.218863f;
    z_4 += features[46] * 0.113720f;
    z_4 += features[47] * 0.140085f;
    z_4 += features[52] * 0.213430f;
    z_4 += features[53] * -0.192858f;
    z_4 += features[54] * 0.175633f;
    z_4 += features[59] * -0.171722f;
    z_4 += features[61] * 0.249332f;
    z_4 += features[62] * -0.137588f;
    float out_4 = 0.229990f * z_4;
    {
        float arg_x = 0.111792f * z_4 + -0.022768f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.106210f * z_4 + 0.031819f) + 1e-6f;
        out_4 += 0.026172f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.164717f * z_4 + -0.072517f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.086272f * z_4 + 0.078488f) + 1e-6f;
        out_4 += -0.046231f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.112768f * z_4 + 0.001608f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.108871f * z_4 + 0.007939f) + 1e-6f;
        out_4 += 0.064289f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.117959f * z_4 + -0.019894f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.111921f * z_4 + 0.031475f) + 1e-6f;
        out_4 += 0.029159f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[4] = out_4;

    // Node 5
    float z_5 = 0.0f;
    z_5 += features[1] * 0.236475f;
    z_5 += features[2] * 0.178520f;
    z_5 += features[3] * 0.136823f;
    z_5 += features[4] * -0.123347f;
    z_5 += features[7] * 0.211552f;
    z_5 += features[8] * -0.168566f;
    z_5 += features[10] * -0.260414f;
    z_5 += features[12] * 0.217216f;
    z_5 += features[15] * -0.162041f;
    z_5 += features[18] * -0.230187f;
    z_5 += features[20] * 0.256594f;
    z_5 += features[23] * -0.186747f;
    z_5 += features[26] * 0.142063f;
    z_5 += features[27] * 0.171235f;
    z_5 += features[31] * 0.137663f;
    z_5 += features[34] * 0.253625f;
    z_5 += features[35] * 0.134738f;
    z_5 += features[41] * -0.144455f;
    z_5 += features[42] * -0.324040f;
    z_5 += features[43] * 0.154510f;
    z_5 += features[44] * 0.159635f;
    z_5 += features[45] * -0.159313f;
    z_5 += features[46] * 0.134186f;
    z_5 += features[47] * -0.200583f;
    z_5 += features[50] * -0.270829f;
    z_5 += features[51] * -0.136302f;
    z_5 += features[53] * 0.211906f;
    z_5 += features[54] * -0.126504f;
    z_5 += features[55] * -0.227597f;
    z_5 += features[56] * 0.192613f;
    z_5 += features[63] * -0.111359f;
    float out_5 = 0.190210f * z_5;
    {
        float arg_x = -0.121445f * z_5 + -0.052624f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.122303f * z_5 + 0.059795f) + 1e-6f;
        out_5 += -0.027520f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.107517f * z_5 + -0.063988f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.109204f * z_5 + 0.068036f) + 1e-6f;
        out_5 += -0.020478f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.138923f * z_5 + -0.108653f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.129104f * z_5 + 0.115906f) + 1e-6f;
        out_5 += -0.004509f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.088142f * z_5 + -0.045209f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.153218f * z_5 + 0.040823f) + 1e-6f;
        out_5 += -0.036944f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[5] = out_5;

    // Node 6
    float z_6 = 0.0f;
    z_6 += features[1] * 0.196125f;
    z_6 += features[5] * 0.181986f;
    z_6 += features[6] * 0.209508f;
    z_6 += features[8] * -0.146130f;
    z_6 += features[9] * -0.159119f;
    z_6 += features[12] * 0.285517f;
    z_6 += features[13] * -0.127454f;
    z_6 += features[14] * 0.170154f;
    z_6 += features[17] * -0.211378f;
    z_6 += features[19] * 0.174497f;
    z_6 += features[20] * -0.275895f;
    z_6 += features[21] * -0.269000f;
    z_6 += features[24] * 0.270769f;
    z_6 += features[25] * -0.174339f;
    z_6 += features[27] * -0.150875f;
    z_6 += features[29] * 0.180117f;
    z_6 += features[32] * 0.373931f;
    z_6 += features[33] * -0.256601f;
    z_6 += features[35] * 0.197310f;
    z_6 += features[37] * 0.227842f;
    z_6 += features[38] * -0.123051f;
    z_6 += features[39] * 0.137963f;
    z_6 += features[43] * 0.231722f;
    z_6 += features[44] * 0.110198f;
    z_6 += features[46] * -0.122629f;
    z_6 += features[47] * -0.141349f;
    z_6 += features[50] * -0.162287f;
    z_6 += features[51] * 0.229879f;
    z_6 += features[52] * -0.139128f;
    z_6 += features[53] * -0.259192f;
    z_6 += features[58] * -0.156148f;
    z_6 += features[60] * 0.229998f;
    z_6 += features[62] * 0.285221f;
    float out_6 = 0.242240f * z_6;
    {
        float arg_x = 0.197822f * z_6 + 0.013552f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.175495f * z_6 + 0.011745f) + 1e-6f;
        out_6 += 0.052766f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.202335f * z_6 + 0.020267f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.173587f * z_6 + 0.007348f) + 1e-6f;
        out_6 += 0.058172f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.065739f * z_6 + -0.054027f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.092989f * z_6 + 0.054880f) + 1e-6f;
        out_6 += -0.022375f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.095427f * z_6 + -0.109447f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.089632f * z_6 + 0.105290f) + 1e-6f;
        out_6 += 0.009248f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[6] = out_6;

    // Node 7
    float z_7 = 0.0f;
    z_7 += features[0] * -0.305444f;
    z_7 += features[1] * -0.132895f;
    z_7 += features[5] * -0.130681f;
    z_7 += features[7] * -0.167810f;
    z_7 += features[8] * -0.229588f;
    z_7 += features[9] * -0.237507f;
    z_7 += features[10] * 0.225455f;
    z_7 += features[16] * 0.228021f;
    z_7 += features[19] * -0.236285f;
    z_7 += features[22] * 0.154126f;
    z_7 += features[23] * 0.147911f;
    z_7 += features[24] * -0.177012f;
    z_7 += features[25] * 0.210510f;
    z_7 += features[26] * -0.115333f;
    z_7 += features[27] * -0.217608f;
    z_7 += features[29] * 0.243987f;
    z_7 += features[30] * -0.235239f;
    z_7 += features[31] * -0.147795f;
    z_7 += features[32] * -0.110899f;
    z_7 += features[33] * 0.272544f;
    z_7 += features[34] * -0.217011f;
    z_7 += features[35] * -0.209991f;
    z_7 += features[36] * -0.170745f;
    z_7 += features[37] * 0.221960f;
    z_7 += features[39] * -0.157702f;
    z_7 += features[40] * -0.144313f;
    z_7 += features[41] * 0.135084f;
    z_7 += features[42] * 0.205183f;
    z_7 += features[44] * 0.134611f;
    z_7 += features[47] * 0.296087f;
    z_7 += features[48] * 0.187620f;
    z_7 += features[49] * 0.142719f;
    z_7 += features[50] * 0.162888f;
    z_7 += features[53] * -0.158585f;
    z_7 += features[54] * 0.228377f;
    z_7 += features[56] * 0.162943f;
    z_7 += features[58] * 0.219230f;
    z_7 += features[61] * 0.255287f;
    z_7 += features[63] * -0.233212f;
    float out_7 = 0.265896f * z_7;
    {
        float arg_x = 0.165944f * z_7 + -0.092157f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.151271f * z_7 + 0.124014f) + 1e-6f;
        out_7 += 0.004930f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.158067f * z_7 + -0.104117f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.167996f * z_7 + 0.128470f) + 1e-6f;
        out_7 += -0.001277f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.041087f * z_7 + -0.044354f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.014649f * z_7 + 0.036685f) + 1e-6f;
        out_7 += -0.048026f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.065047f * z_7 + -0.019475f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.071790f * z_7 + 0.009574f) + 1e-6f;
        out_7 += -0.061651f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[7] = out_7;

    // Node 8
    float z_8 = 0.0f;
    z_8 += features[0] * 0.273831f;
    z_8 += features[4] * 0.129341f;
    z_8 += features[8] * 0.217914f;
    z_8 += features[11] * 0.207950f;
    z_8 += features[12] * 0.240999f;
    z_8 += features[15] * 0.168710f;
    z_8 += features[16] * -0.119060f;
    z_8 += features[17] * 0.156273f;
    z_8 += features[21] * 0.144803f;
    z_8 += features[23] * 0.179576f;
    z_8 += features[24] * 0.136695f;
    z_8 += features[25] * -0.150439f;
    z_8 += features[26] * 0.124840f;
    z_8 += features[30] * 0.207938f;
    z_8 += features[31] * -0.223724f;
    z_8 += features[32] * -0.238114f;
    z_8 += features[34] * -0.192425f;
    z_8 += features[35] * -0.268442f;
    z_8 += features[37] * -0.178398f;
    z_8 += features[38] * -0.226978f;
    z_8 += features[40] * -0.162222f;
    z_8 += features[41] * -0.161979f;
    z_8 += features[43] * -0.131074f;
    z_8 += features[44] * -0.136343f;
    z_8 += features[45] * 0.316543f;
    z_8 += features[46] * 0.180398f;
    z_8 += features[47] * 0.237833f;
    z_8 += features[48] * -0.164293f;
    z_8 += features[50] * 0.204397f;
    z_8 += features[56] * 0.147486f;
    z_8 += features[59] * -0.182993f;
    z_8 += features[62] * -0.130156f;
    z_8 += features[63] * -0.249653f;
    float out_8 = 0.236921f * z_8;
    {
        float arg_x = -0.157648f * z_8 + -0.021611f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.142680f * z_8 + 0.035137f) + 1e-6f;
        out_8 += -0.020594f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.035109f * z_8 + -0.163038f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.078213f * z_8 + 0.162813f) + 1e-6f;
        out_8 += 0.021016f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.171739f * z_8 + -0.019288f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.133076f * z_8 + 0.038330f) + 1e-6f;
        out_8 += -0.016694f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.171791f * z_8 + -0.010856f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.134634f * z_8 + 0.030912f) + 1e-6f;
        out_8 += -0.035084f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[8] = out_8;

    // Node 9
    float z_9 = 0.0f;
    z_9 += features[2] * 0.196643f;
    z_9 += features[4] * -0.170981f;
    z_9 += features[8] * -0.173035f;
    z_9 += features[9] * -0.122587f;
    z_9 += features[16] * -0.173369f;
    z_9 += features[18] * -0.206706f;
    z_9 += features[20] * 0.114480f;
    z_9 += features[22] * 0.171177f;
    z_9 += features[27] * -0.156867f;
    z_9 += features[28] * -0.150664f;
    z_9 += features[34] * 0.226243f;
    z_9 += features[36] * 0.155986f;
    z_9 += features[39] * 0.141479f;
    z_9 += features[40] * 0.146207f;
    z_9 += features[43] * 0.165782f;
    z_9 += features[49] * 0.144719f;
    z_9 += features[50] * -0.110400f;
    z_9 += features[51] * 0.162022f;
    z_9 += features[52] * -0.152561f;
    z_9 += features[53] * -0.137272f;
    z_9 += features[54] * -0.185555f;
    z_9 += features[55] * 0.187658f;
    z_9 += features[56] * 0.225053f;
    z_9 += features[58] * 0.229694f;
    z_9 += features[60] * -0.230907f;
    float out_9 = 0.184201f * z_9;
    {
        float arg_x = -0.020115f * z_9 + -0.135347f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.002687f * z_9 + 0.134795f) + 1e-6f;
        out_9 += -0.013737f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.016455f * z_9 + -0.145218f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.048220f * z_9 + 0.155367f) + 1e-6f;
        out_9 += -0.009188f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.069123f * z_9 + -0.065664f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.092078f * z_9 + 0.060519f) + 1e-6f;
        out_9 += 0.009070f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.015070f * z_9 + -0.137021f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.039645f * z_9 + 0.138117f) + 1e-6f;
        out_9 += -0.006561f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[9] = out_9;

    // Node 10
    float z_10 = 0.0f;
    z_10 += features[1] * -0.202513f;
    z_10 += features[4] * -0.161215f;
    z_10 += features[5] * -0.218165f;
    z_10 += features[7] * 0.231344f;
    z_10 += features[10] * -0.175820f;
    z_10 += features[11] * -0.181414f;
    z_10 += features[13] * -0.131839f;
    z_10 += features[15] * -0.170198f;
    z_10 += features[16] * 0.262696f;
    z_10 += features[20] * -0.139716f;
    z_10 += features[21] * 0.112833f;
    z_10 += features[26] * -0.243937f;
    z_10 += features[29] * 0.230161f;
    z_10 += features[33] * -0.161406f;
    z_10 += features[35] * 0.155427f;
    z_10 += features[36] * 0.181516f;
    z_10 += features[39] * -0.121432f;
    z_10 += features[41] * 0.261797f;
    z_10 += features[45] * -0.196769f;
    z_10 += features[47] * 0.123154f;
    z_10 += features[49] * -0.180623f;
    z_10 += features[52] * -0.153400f;
    z_10 += features[54] * 0.293287f;
    z_10 += features[58] * -0.168009f;
    z_10 += features[59] * -0.124020f;
    z_10 += features[60] * 0.167137f;
    z_10 += features[62] * 0.238289f;
    z_10 += features[63] * 0.131115f;
    float out_10 = 0.185186f * z_10;
    {
        float arg_x = 0.115590f * z_10 + -0.022100f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.150144f * z_10 + 0.019935f) + 1e-6f;
        out_10 += 0.049764f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.031349f * z_10 + -0.083726f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.033100f * z_10 + 0.081755f) + 1e-6f;
        out_10 += 0.027819f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.166726f * z_10 + 0.016157f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.108358f * z_10 + -0.010121f) + 1e-6f;
        out_10 += 0.057241f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.064524f * z_10 + -0.068638f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.043160f * z_10 + 0.066935f) + 1e-6f;
        out_10 += 0.035700f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[10] = out_10;

    // Node 11
    float z_11 = 0.0f;
    z_11 += features[0] * 0.183835f;
    z_11 += features[6] * 0.205272f;
    z_11 += features[9] * 0.187703f;
    z_11 += features[12] * -0.118691f;
    z_11 += features[13] * -0.167533f;
    z_11 += features[14] * 0.152201f;
    z_11 += features[18] * 0.143289f;
    z_11 += features[21] * -0.180448f;
    z_11 += features[22] * -0.175904f;
    z_11 += features[24] * -0.134941f;
    z_11 += features[29] * -0.230715f;
    z_11 += features[31] * -0.169779f;
    z_11 += features[37] * -0.149094f;
    z_11 += features[38] * -0.221324f;
    z_11 += features[45] * 0.241879f;
    z_11 += features[46] * 0.119775f;
    z_11 += features[49] * -0.160010f;
    z_11 += features[50] * 0.184246f;
    z_11 += features[51] * -0.120482f;
    z_11 += features[52] * 0.177940f;
    z_11 += features[55] * -0.138072f;
    z_11 += features[56] * -0.234796f;
    z_11 += features[57] * 0.120675f;
    z_11 += features[58] * -0.227763f;
    z_11 += features[59] * -0.201142f;
    z_11 += features[62] * -0.158869f;
    float out_11 = 0.180136f * z_11;
    {
        float arg_x = 0.028212f * z_11 + -0.015707f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.052632f * z_11 + 0.013404f) + 1e-6f;
        out_11 += -0.036024f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.076217f * z_11 + -0.076524f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.092179f * z_11 + 0.079970f) + 1e-6f;
        out_11 += -0.019180f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.093335f * z_11 + 0.047607f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.098566f * z_11 + -0.047498f) + 1e-6f;
        out_11 += -0.101251f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.108807f * z_11 + -0.068687f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.051507f * z_11 + 0.094971f) + 1e-6f;
        out_11 += -0.012826f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[11] = out_11;

    // Node 12
    float z_12 = 0.0f;
    z_12 += features[1] * -0.174373f;
    z_12 += features[3] * -0.228920f;
    z_12 += features[4] * -0.287457f;
    z_12 += features[7] * 0.258187f;
    z_12 += features[9] * 0.148134f;
    z_12 += features[10] * 0.264841f;
    z_12 += features[12] * 0.169778f;
    z_12 += features[13] * 0.113137f;
    z_12 += features[16] * 0.268116f;
    z_12 += features[17] * -0.170757f;
    z_12 += features[20] * -0.140260f;
    z_12 += features[24] * -0.130200f;
    z_12 += features[26] * -0.211502f;
    z_12 += features[27] * 0.149563f;
    z_12 += features[31] * 0.120128f;
    z_12 += features[32] * 0.283632f;
    z_12 += features[34] * 0.157254f;
    z_12 += features[35] * 0.260319f;
    z_12 += features[36] * -0.157405f;
    z_12 += features[38] * 0.297263f;
    z_12 += features[42] * -0.231691f;
    z_12 += features[44] * 0.186014f;
    z_12 += features[45] * -0.216202f;
    z_12 += features[48] * 0.181564f;
    z_12 += features[49] * -0.266008f;
    z_12 += features[50] * -0.141239f;
    z_12 += features[51] * -0.109932f;
    z_12 += features[52] * 0.231265f;
    z_12 += features[54] * -0.200614f;
    z_12 += features[55] * 0.126401f;
    z_12 += features[56] * 0.184837f;
    z_12 += features[58] * 0.125625f;
    z_12 += features[60] * 0.297494f;
    z_12 += features[61] * -0.120926f;
    float out_12 = 0.257065f * z_12;
    {
        float arg_x = -0.192870f * z_12 + 0.022717f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.187771f * z_12 + -0.007203f) + 1e-6f;
        out_12 += -0.077351f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.188458f * z_12 + -0.056480f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.117866f * z_12 + 0.066461f) + 1e-6f;
        out_12 += -0.042376f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.160733f * z_12 + -0.027304f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.150589f * z_12 + 0.063951f) + 1e-6f;
        out_12 += 0.035520f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.154353f * z_12 + -0.079410f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.189369f * z_12 + 0.105550f) + 1e-6f;
        out_12 += 0.025761f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[12] = out_12;

    // Node 13
    float z_13 = 0.0f;
    z_13 += features[0] * -0.189630f;
    z_13 += features[2] * 0.147871f;
    z_13 += features[3] * 0.131471f;
    z_13 += features[5] * -0.189030f;
    z_13 += features[6] * -0.134774f;
    z_13 += features[10] * -0.166479f;
    z_13 += features[11] * -0.158066f;
    z_13 += features[12] * -0.295738f;
    z_13 += features[13] * -0.147943f;
    z_13 += features[14] * 0.168014f;
    z_13 += features[16] * 0.222813f;
    z_13 += features[20] * 0.169442f;
    z_13 += features[21] * -0.136932f;
    z_13 += features[22] * 0.121849f;
    z_13 += features[24] * -0.249350f;
    z_13 += features[25] * 0.259011f;
    z_13 += features[27] * 0.166763f;
    z_13 += features[32] * -0.139033f;
    z_13 += features[33] * 0.279582f;
    z_13 += features[36] * 0.187201f;
    z_13 += features[37] * -0.119523f;
    z_13 += features[38] * 0.237439f;
    z_13 += features[39] * 0.257607f;
    z_13 += features[40] * 0.357785f;
    z_13 += features[41] * 0.286304f;
    z_13 += features[43] * -0.240416f;
    z_13 += features[45] * 0.145698f;
    z_13 += features[47] * -0.172586f;
    z_13 += features[50] * 0.183389f;
    z_13 += features[51] * 0.234407f;
    z_13 += features[53] * -0.187382f;
    z_13 += features[59] * 0.111470f;
    z_13 += features[60] * -0.204577f;
    z_13 += features[61] * -0.111514f;
    float out_13 = 0.264771f * z_13;
    {
        float arg_x = 0.101743f * z_13 + -0.071295f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.103496f * z_13 + 0.077514f) + 1e-6f;
        out_13 += 0.023189f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.331948f * z_13 + 0.080168f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.228262f * z_13 + -0.018258f) + 1e-6f;
        out_13 += -0.112730f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.229892f * z_13 + -0.051366f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.181378f * z_13 + 0.068811f) + 1e-6f;
        out_13 += -0.041968f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.007198f * z_13 + -0.148967f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.049924f * z_13 + 0.150837f) + 1e-6f;
        out_13 += 0.006716f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[13] = out_13;

    // Node 14
    float z_14 = 0.0f;
    z_14 += features[0] * -0.194687f;
    z_14 += features[2] * 0.140065f;
    z_14 += features[7] * 0.129031f;
    z_14 += features[8] * -0.167522f;
    z_14 += features[11] * -0.186897f;
    z_14 += features[13] * 0.246799f;
    z_14 += features[15] * -0.133880f;
    z_14 += features[20] * 0.158870f;
    z_14 += features[22] * -0.155192f;
    z_14 += features[23] * -0.202426f;
    z_14 += features[24] * -0.155582f;
    z_14 += features[25] * 0.156403f;
    z_14 += features[26] * -0.117889f;
    z_14 += features[27] * 0.146661f;
    z_14 += features[32] * -0.157552f;
    z_14 += features[34] * 0.129664f;
    z_14 += features[35] * 0.194711f;
    z_14 += features[36] * -0.227323f;
    z_14 += features[38] * 0.252153f;
    z_14 += features[44] * 0.168608f;
    z_14 += features[46] * 0.144778f;
    z_14 += features[50] * -0.157921f;
    z_14 += features[54] * -0.232067f;
    z_14 += features[56] * 0.140553f;
    z_14 += features[58] * 0.226113f;
    z_14 += features[59] * 0.162437f;
    z_14 += features[63] * 0.146167f;
    float out_14 = 0.214271f * z_14;
    {
        float arg_x = 0.076841f * z_14 + 0.021111f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.057451f * z_14 + -0.009774f) + 1e-6f;
        out_14 += 0.059200f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.138895f * z_14 + -0.060511f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.146182f * z_14 + 0.065634f) + 1e-6f;
        out_14 += 0.013228f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.105407f * z_14 + -0.058319f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.122783f * z_14 + 0.061944f) + 1e-6f;
        out_14 += -0.013092f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.168588f * z_14 + -0.061248f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.106575f * z_14 + 0.096982f) + 1e-6f;
        out_14 += -0.003795f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[14] = out_14;

    // Node 15
    float z_15 = 0.0f;
    z_15 += features[2] * -0.138393f;
    z_15 += features[3] * 0.146297f;
    z_15 += features[8] * -0.208350f;
    z_15 += features[10] * 0.256220f;
    z_15 += features[14] * 0.183816f;
    z_15 += features[15] * -0.111126f;
    z_15 += features[17] * -0.154878f;
    z_15 += features[19] * 0.186365f;
    z_15 += features[22] * -0.147385f;
    z_15 += features[24] * -0.162712f;
    z_15 += features[25] * 0.238442f;
    z_15 += features[29] * -0.160137f;
    z_15 += features[31] * -0.173869f;
    z_15 += features[33] * 0.204144f;
    z_15 += features[34] * 0.218139f;
    z_15 += features[36] * -0.180856f;
    z_15 += features[37] * -0.142852f;
    z_15 += features[40] * 0.137475f;
    z_15 += features[41] * -0.162581f;
    z_15 += features[42] * -0.175216f;
    z_15 += features[44] * 0.117710f;
    z_15 += features[45] * 0.149820f;
    z_15 += features[46] * -0.129160f;
    z_15 += features[48] * -0.146657f;
    z_15 += features[51] * -0.168789f;
    z_15 += features[52] * 0.160115f;
    z_15 += features[54] * -0.327175f;
    z_15 += features[55] * 0.120615f;
    z_15 += features[56] * 0.232843f;
    z_15 += features[60] * -0.251471f;
    z_15 += features[62] * -0.239197f;
    z_15 += features[63] * 0.186037f;
    float out_15 = 0.198925f * z_15;
    {
        float arg_x = -0.078154f * z_15 + 0.014995f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.084136f * z_15 + -0.016181f) + 1e-6f;
        out_15 += -0.067170f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.100171f * z_15 + -0.039822f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.109677f * z_15 + 0.042167f) + 1e-6f;
        out_15 += -0.015682f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.119993f * z_15 + -0.035248f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.091076f * z_15 + 0.043895f) + 1e-6f;
        out_15 += -0.025209f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.111976f * z_15 + -0.039285f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.135736f * z_15 + 0.042670f) + 1e-6f;
        out_15 += 0.005599f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[15] = out_15;

    // Node 16
    float z_16 = 0.0f;
    z_16 += features[1] * 0.201924f;
    z_16 += features[4] * 0.149138f;
    z_16 += features[7] * -0.161079f;
    z_16 += features[8] * -0.224845f;
    z_16 += features[10] * 0.165254f;
    z_16 += features[11] * 0.288780f;
    z_16 += features[17] * -0.209174f;
    z_16 += features[19] * -0.200785f;
    z_16 += features[20] * 0.146684f;
    z_16 += features[21] * 0.226770f;
    z_16 += features[23] * 0.207006f;
    z_16 += features[24] * -0.217452f;
    z_16 += features[25] * 0.217494f;
    z_16 += features[27] * -0.191923f;
    z_16 += features[29] * -0.171329f;
    z_16 += features[30] * -0.122278f;
    z_16 += features[32] * 0.114542f;
    z_16 += features[33] * 0.123695f;
    z_16 += features[39] * 0.135335f;
    z_16 += features[42] * -0.220358f;
    z_16 += features[43] * 0.133618f;
    z_16 += features[44] * 0.146094f;
    z_16 += features[46] * 0.182481f;
    z_16 += features[47] * 0.116042f;
    z_16 += features[48] * -0.242212f;
    z_16 += features[49] * 0.119758f;
    z_16 += features[52] * -0.116523f;
    z_16 += features[53] * -0.131487f;
    z_16 += features[54] * -0.181471f;
    z_16 += features[56] * 0.110709f;
    z_16 += features[58] * 0.234756f;
    z_16 += features[59] * 0.232138f;
    z_16 += features[60] * -0.140644f;
    z_16 += features[62] * -0.207977f;
    z_16 += features[63] * -0.151208f;
    float out_16 = 0.241294f * z_16;
    {
        float arg_x = 0.167711f * z_16 + -0.053858f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.168021f * z_16 + 0.064540f) + 1e-6f;
        out_16 += 0.020764f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.096336f * z_16 + 0.031067f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.130613f * z_16 + -0.033036f) + 1e-6f;
        out_16 += -0.081938f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.133474f * z_16 + -0.064487f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.161761f * z_16 + 0.068529f) + 1e-6f;
        out_16 += 0.013606f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.019472f * z_16 + -0.114809f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.000503f * z_16 + 0.108763f) + 1e-6f;
        out_16 += -0.017538f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[16] = out_16;

    // Node 17
    float z_17 = 0.0f;
    z_17 += features[1] * -0.183330f;
    z_17 += features[2] * -0.192776f;
    z_17 += features[10] * 0.130206f;
    z_17 += features[11] * 0.225374f;
    z_17 += features[15] * -0.209922f;
    z_17 += features[16] * 0.157101f;
    z_17 += features[17] * -0.280394f;
    z_17 += features[20] * -0.153430f;
    z_17 += features[27] * -0.177380f;
    z_17 += features[30] * -0.176228f;
    z_17 += features[32] * 0.327840f;
    z_17 += features[36] * -0.144126f;
    z_17 += features[38] * 0.152117f;
    z_17 += features[40] * -0.325358f;
    z_17 += features[41] * -0.256254f;
    z_17 += features[42] * -0.224116f;
    z_17 += features[45] * -0.158779f;
    z_17 += features[47] * 0.117228f;
    z_17 += features[52] * 0.230391f;
    z_17 += features[53] * -0.167603f;
    z_17 += features[54] * 0.112273f;
    z_17 += features[55] * 0.165062f;
    z_17 += features[56] * -0.112817f;
    z_17 += features[57] * -0.123464f;
    z_17 += features[58] * 0.150363f;
    z_17 += features[61] * 0.238656f;
    float out_17 = 0.153462f * z_17;
    {
        float arg_x = -0.019438f * z_17 + -0.146046f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.015251f * z_17 + 0.147060f) + 1e-6f;
        out_17 += 0.000271f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.086326f * z_17 + -0.100986f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.091823f * z_17 + 0.106547f) + 1e-6f;
        out_17 += -0.004402f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.048916f * z_17 + -0.024706f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.061757f * z_17 + 0.021892f) + 1e-6f;
        out_17 += -0.034076f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.072145f * z_17 + -0.047660f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.049732f * z_17 + 0.050985f) + 1e-6f;
        out_17 += -0.012902f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[17] = out_17;

    // Node 18
    float z_18 = 0.0f;
    z_18 += features[0] * 0.134478f;
    z_18 += features[1] * 0.201848f;
    z_18 += features[4] * 0.135843f;
    z_18 += features[6] * -0.118989f;
    z_18 += features[7] * -0.325501f;
    z_18 += features[8] * 0.125475f;
    z_18 += features[10] * -0.138795f;
    z_18 += features[11] * 0.112032f;
    z_18 += features[12] * -0.175414f;
    z_18 += features[13] * -0.185682f;
    z_18 += features[15] * 0.145542f;
    z_18 += features[16] * -0.145085f;
    z_18 += features[17] * 0.292993f;
    z_18 += features[20] * 0.197302f;
    z_18 += features[21] * -0.184407f;
    z_18 += features[23] * 0.167909f;
    z_18 += features[26] * 0.115090f;
    z_18 += features[29] * -0.318187f;
    z_18 += features[38] * -0.278082f;
    z_18 += features[39] * -0.194313f;
    z_18 += features[40] * 0.114875f;
    z_18 += features[41] * -0.287428f;
    z_18 += features[42] * 0.204098f;
    z_18 += features[43] * -0.245098f;
    z_18 += features[45] * 0.260428f;
    z_18 += features[48] * -0.277326f;
    z_18 += features[49] * 0.111966f;
    z_18 += features[50] * 0.156273f;
    z_18 += features[53] * 0.242927f;
    z_18 += features[55] * -0.181069f;
    z_18 += features[60] * -0.246720f;
    z_18 += features[61] * -0.194825f;
    z_18 += features[62] * -0.207435f;
    z_18 += features[63] * 0.115736f;
    float out_18 = 0.278254f * z_18;
    {
        float arg_x = -0.189864f * z_18 + 0.027067f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.193871f * z_18 + 0.008901f) + 1e-6f;
        out_18 += -0.075040f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.195791f * z_18 + 0.039434f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.169956f * z_18 + 0.005343f) + 1e-6f;
        out_18 += -0.092369f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.049524f * z_18 + -0.193709f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.052184f * z_18 + 0.190462f) + 1e-6f;
        out_18 += 0.014635f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.069244f * z_18 + -0.172480f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.023075f * z_18 + 0.178124f) + 1e-6f;
        out_18 += 0.011412f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[18] = out_18;

    // Node 19
    float z_19 = 0.0f;
    z_19 += features[2] * -0.178907f;
    z_19 += features[4] * 0.186698f;
    z_19 += features[5] * -0.138364f;
    z_19 += features[6] * 0.143635f;
    z_19 += features[9] * 0.176886f;
    z_19 += features[14] * -0.123846f;
    z_19 += features[19] * 0.109829f;
    z_19 += features[20] * -0.146931f;
    z_19 += features[24] * -0.160392f;
    z_19 += features[27] * 0.172803f;
    z_19 += features[28] * 0.189158f;
    z_19 += features[31] * -0.172914f;
    z_19 += features[39] * -0.200589f;
    z_19 += features[45] * 0.244082f;
    z_19 += features[51] * -0.165403f;
    z_19 += features[54] * 0.191590f;
    z_19 += features[57] * 0.152063f;
    z_19 += features[62] * -0.143945f;
    z_19 += features[63] * 0.144427f;
    float out_19 = 0.141339f * z_19;
    {
        float arg_x = 0.038430f * z_19 + -0.045021f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.043588f * z_19 + 0.045144f) + 1e-6f;
        out_19 += -0.003081f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.049475f * z_19 + -0.032004f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.033072f * z_19 + 0.032332f) + 1e-6f;
        out_19 += 0.012835f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.036337f * z_19 + -0.038073f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.051381f * z_19 + 0.037014f) + 1e-6f;
        out_19 += 0.005191f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.045401f * z_19 + -0.118632f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.085519f * z_19 + 0.116124f) + 1e-6f;
        out_19 += -0.030194f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[19] = out_19;

    // Node 20
    float z_20 = 0.0f;
    z_20 += features[0] * 0.111114f;
    z_20 += features[1] * -0.212806f;
    z_20 += features[3] * 0.142208f;
    z_20 += features[4] * 0.196680f;
    z_20 += features[7] * 0.224862f;
    z_20 += features[16] * 0.169559f;
    z_20 += features[18] * 0.116006f;
    z_20 += features[20] * -0.157071f;
    z_20 += features[21] * 0.163954f;
    z_20 += features[22] * 0.238351f;
    z_20 += features[23] * -0.161070f;
    z_20 += features[24] * -0.262770f;
    z_20 += features[25] * 0.196623f;
    z_20 += features[28] * 0.142353f;
    z_20 += features[29] * -0.200945f;
    z_20 += features[30] * 0.183599f;
    z_20 += features[32] * -0.266704f;
    z_20 += features[33] * 0.125066f;
    z_20 += features[36] * -0.172188f;
    z_20 += features[37] * -0.127931f;
    z_20 += features[38] * 0.172974f;
    z_20 += features[40] * -0.182563f;
    z_20 += features[43] * -0.155037f;
    z_20 += features[46] * 0.210065f;
    z_20 += features[49] * -0.257646f;
    z_20 += features[51] * -0.252551f;
    z_20 += features[53] * -0.126808f;
    z_20 += features[55] * 0.113139f;
    z_20 += features[56] * 0.268787f;
    z_20 += features[58] * 0.284016f;
    z_20 += features[59] * -0.126013f;
    z_20 += features[63] * -0.164684f;
    float out_20 = 0.203013f * z_20;
    {
        float arg_x = -0.092610f * z_20 + -0.015274f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.110409f * z_20 + 0.004617f) + 1e-6f;
        out_20 += -0.032596f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.139541f * z_20 + -0.051700f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.123270f * z_20 + 0.045677f) + 1e-6f;
        out_20 += -0.013097f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.153362f * z_20 + -0.035807f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.125854f * z_20 + 0.032787f) + 1e-6f;
        out_20 += -0.019090f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.197521f * z_20 + -0.021458f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.098998f * z_20 + 0.027921f) + 1e-6f;
        out_20 += -0.026210f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[20] = out_20;

    // Node 21
    float z_21 = 0.0f;
    z_21 += features[0] * -0.196441f;
    z_21 += features[2] * 0.115765f;
    z_21 += features[4] * 0.128024f;
    z_21 += features[6] * -0.194148f;
    z_21 += features[10] * -0.193397f;
    z_21 += features[11] * -0.261773f;
    z_21 += features[24] * -0.240568f;
    z_21 += features[26] * -0.234181f;
    z_21 += features[27] * 0.256927f;
    z_21 += features[28] * -0.183970f;
    z_21 += features[30] * 0.227618f;
    z_21 += features[31] * 0.194067f;
    z_21 += features[32] * -0.121975f;
    z_21 += features[35] * 0.241684f;
    z_21 += features[36] * 0.138344f;
    z_21 += features[40] * 0.297781f;
    z_21 += features[41] * 0.272581f;
    z_21 += features[42] * -0.154100f;
    z_21 += features[45] * -0.202640f;
    z_21 += features[47] * 0.112921f;
    z_21 += features[48] * -0.133747f;
    z_21 += features[49] * -0.204107f;
    z_21 += features[50] * 0.149370f;
    z_21 += features[53] * 0.178225f;
    z_21 += features[55] * 0.200477f;
    z_21 += features[57] * 0.146898f;
    z_21 += features[58] * 0.218744f;
    z_21 += features[59] * 0.234031f;
    z_21 += features[60] * 0.170671f;
    z_21 += features[61] * -0.195303f;
    z_21 += features[62] * 0.227570f;
    float out_21 = 0.232642f * z_21;
    {
        float arg_x = 0.065256f * z_21 + -0.063437f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.069253f * z_21 + 0.062961f) + 1e-6f;
        out_21 += 0.047182f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.077751f * z_21 + -0.106078f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.016753f * z_21 + 0.111428f) + 1e-6f;
        out_21 += 0.021227f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.114208f * z_21 + -0.020245f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.107766f * z_21 + 0.028873f) + 1e-6f;
        out_21 += -0.026453f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.083009f * z_21 + -0.071736f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.111926f * z_21 + 0.071150f) + 1e-6f;
        out_21 += -0.001831f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[21] = out_21;

    // Node 22
    float z_22 = 0.0f;
    z_22 += features[2] * -0.140647f;
    z_22 += features[3] * 0.141259f;
    z_22 += features[7] * -0.274299f;
    z_22 += features[11] * 0.211410f;
    z_22 += features[14] * -0.199633f;
    z_22 += features[16] * -0.154862f;
    z_22 += features[17] * 0.172539f;
    z_22 += features[19] * -0.121254f;
    z_22 += features[20] * -0.160195f;
    z_22 += features[23] * 0.206677f;
    z_22 += features[25] * -0.158372f;
    z_22 += features[26] * 0.112687f;
    z_22 += features[27] * -0.279422f;
    z_22 += features[28] * 0.157590f;
    z_22 += features[30] * 0.180868f;
    z_22 += features[31] * -0.141034f;
    z_22 += features[32] * -0.274180f;
    z_22 += features[34] * -0.219324f;
    z_22 += features[35] * -0.221998f;
    z_22 += features[36] * 0.228411f;
    z_22 += features[45] * 0.240678f;
    z_22 += features[47] * 0.227188f;
    z_22 += features[52] * 0.186279f;
    z_22 += features[56] * -0.138352f;
    z_22 += features[58] * -0.246478f;
    z_22 += features[59] * -0.196038f;
    z_22 += features[62] * 0.206960f;
    float out_22 = 0.226094f * z_22;
    {
        float arg_x = 0.047927f * z_22 + -0.173654f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.048654f * z_22 + 0.179947f) + 1e-6f;
        out_22 += 0.006816f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.117130f * z_22 + -0.017561f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.101322f * z_22 + 0.043466f) + 1e-6f;
        out_22 += 0.047075f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.057230f * z_22 + -0.168718f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.067233f * z_22 + 0.173992f) + 1e-6f;
        out_22 += 0.009696f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.079588f * z_22 + -0.127849f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.095926f * z_22 + 0.136602f) + 1e-6f;
        out_22 += 0.019736f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[22] = out_22;

    // Node 23
    float z_23 = 0.0f;
    z_23 += features[3] * -0.121271f;
    z_23 += features[4] * 0.135358f;
    z_23 += features[8] * 0.221396f;
    z_23 += features[9] * -0.134552f;
    z_23 += features[10] * -0.141723f;
    z_23 += features[11] * -0.253612f;
    z_23 += features[12] * -0.181132f;
    z_23 += features[13] * -0.173152f;
    z_23 += features[15] * 0.178506f;
    z_23 += features[16] * -0.115415f;
    z_23 += features[18] * 0.241687f;
    z_23 += features[22] * 0.139691f;
    z_23 += features[29] * 0.164420f;
    z_23 += features[34] * -0.204201f;
    z_23 += features[36] * 0.142613f;
    z_23 += features[38] * -0.125831f;
    z_23 += features[41] * 0.274770f;
    z_23 += features[42] * 0.128193f;
    z_23 += features[43] * -0.240558f;
    z_23 += features[44] * -0.167356f;
    z_23 += features[45] * 0.234440f;
    z_23 += features[48] * 0.160143f;
    z_23 += features[49] * -0.236037f;
    z_23 += features[54] * 0.147243f;
    z_23 += features[58] * -0.183886f;
    z_23 += features[59] * -0.162460f;
    z_23 += features[60] * 0.275197f;
    z_23 += features[61] * 0.118987f;
    z_23 += features[62] * 0.123011f;
    z_23 += features[63] * 0.127586f;
    float out_23 = 0.205759f * z_23;
    {
        float arg_x = 0.120276f * z_23 + -0.065550f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.124094f * z_23 + 0.073342f) + 1e-6f;
        out_23 += -0.010758f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.085566f * z_23 + 0.037048f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.044007f * z_23 + -0.035619f) + 1e-6f;
        out_23 += -0.041263f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.121109f * z_23 + -0.067026f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.116210f * z_23 + 0.079563f) + 1e-6f;
        out_23 += -0.010624f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.113970f * z_23 + -0.102238f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.157925f * z_23 + 0.109970f) + 1e-6f;
        out_23 += -0.003866f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[23] = out_23;

    // Node 24
    float z_24 = 0.0f;
    z_24 += features[5] * -0.138851f;
    z_24 += features[8] * -0.202449f;
    z_24 += features[10] * 0.127292f;
    z_24 += features[17] * -0.185468f;
    z_24 += features[18] * -0.114591f;
    z_24 += features[19] * 0.142475f;
    z_24 += features[20] * 0.122040f;
    z_24 += features[21] * -0.241882f;
    z_24 += features[24] * -0.157396f;
    z_24 += features[25] * 0.173098f;
    z_24 += features[30] * -0.174322f;
    z_24 += features[32] * 0.212088f;
    z_24 += features[33] * 0.203985f;
    z_24 += features[35] * 0.146876f;
    z_24 += features[38] * 0.225380f;
    z_24 += features[40] * 0.172796f;
    z_24 += features[41] * -0.230109f;
    z_24 += features[42] * -0.113435f;
    z_24 += features[44] * 0.194496f;
    z_24 += features[45] * -0.144886f;
    z_24 += features[48] * 0.149389f;
    z_24 += features[50] * -0.207234f;
    z_24 += features[53] * -0.161241f;
    z_24 += features[54] * -0.233866f;
    z_24 += features[57] * -0.209745f;
    z_24 += features[58] * 0.240168f;
    z_24 += features[60] * -0.209684f;
    z_24 += features[62] * -0.111695f;
    z_24 += features[63] * 0.191017f;
    float out_24 = 0.219332f * z_24;
    {
        float arg_x = 0.098438f * z_24 + -0.124487f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.066596f * z_24 + 0.131498f) + 1e-6f;
        out_24 += -0.016671f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.067483f * z_24 + -0.093372f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.061111f * z_24 + 0.092604f) + 1e-6f;
        out_24 += -0.022680f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.106103f * z_24 + -0.012118f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.109322f * z_24 + 0.018756f) + 1e-6f;
        out_24 += 0.049515f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.120329f * z_24 + -0.045099f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.066598f * z_24 + 0.061796f) + 1e-6f;
        out_24 += 0.001766f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[24] = out_24;

    // Node 25
    float z_25 = 0.0f;
    z_25 += features[0] * -0.243029f;
    z_25 += features[1] * -0.188839f;
    z_25 += features[2] * -0.130067f;
    z_25 += features[3] * -0.150200f;
    z_25 += features[4] * -0.173487f;
    z_25 += features[5] * -0.279685f;
    z_25 += features[7] * 0.130377f;
    z_25 += features[8] * -0.141533f;
    z_25 += features[11] * -0.187080f;
    z_25 += features[14] * 0.140641f;
    z_25 += features[16] * 0.299167f;
    z_25 += features[17] * -0.218515f;
    z_25 += features[19] * 0.196449f;
    z_25 += features[21] * -0.201869f;
    z_25 += features[23] * 0.191734f;
    z_25 += features[24] * 0.200071f;
    z_25 += features[29] * 0.227789f;
    z_25 += features[32] * 0.176349f;
    z_25 += features[33] * 0.121174f;
    z_25 += features[34] * -0.230435f;
    z_25 += features[37] * 0.201369f;
    z_25 += features[38] * 0.285638f;
    z_25 += features[40] * 0.181793f;
    z_25 += features[41] * 0.226497f;
    z_25 += features[42] * 0.310588f;
    z_25 += features[44] * 0.113198f;
    z_25 += features[46] * -0.161865f;
    z_25 += features[49] * 0.239679f;
    z_25 += features[51] * 0.175353f;
    z_25 += features[53] * -0.134604f;
    z_25 += features[54] * 0.201346f;
    z_25 += features[55] * 0.188433f;
    z_25 += features[56] * -0.219944f;
    z_25 += features[59] * 0.206398f;
    z_25 += features[62] * 0.206470f;
    float out_25 = 0.269697f * z_25;
    {
        float arg_x = 0.272718f * z_25 + 0.088945f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.176011f * z_25 + -0.033680f) + 1e-6f;
        out_25 += 0.101495f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.163762f * z_25 + -0.050522f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.126731f * z_25 + 0.080534f) + 1e-6f;
        out_25 += -0.031967f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.211892f * z_25 + 0.053392f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.188320f * z_25 + -0.027327f) + 1e-6f;
        out_25 += 0.074266f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.237859f * z_25 + 0.071464f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.145529f * z_25 + -0.027820f) + 1e-6f;
        out_25 += 0.088439f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[25] = out_25;

    // Node 26
    float z_26 = 0.0f;
    z_26 += features[0] * -0.155860f;
    z_26 += features[3] * -0.120711f;
    z_26 += features[4] * -0.151867f;
    z_26 += features[5] * -0.162502f;
    z_26 += features[8] * -0.239903f;
    z_26 += features[9] * 0.128503f;
    z_26 += features[10] * 0.157478f;
    z_26 += features[11] * -0.167381f;
    z_26 += features[12] * 0.213909f;
    z_26 += features[13] * 0.148699f;
    z_26 += features[14] * 0.161826f;
    z_26 += features[15] * -0.145309f;
    z_26 += features[17] * -0.146135f;
    z_26 += features[18] * -0.143153f;
    z_26 += features[19] * 0.215548f;
    z_26 += features[21] * -0.299018f;
    z_26 += features[22] * -0.256812f;
    z_26 += features[23] * 0.134430f;
    z_26 += features[28] * 0.164602f;
    z_26 += features[29] * 0.127615f;
    z_26 += features[30] * -0.236577f;
    z_26 += features[32] * 0.282152f;
    z_26 += features[33] * 0.144821f;
    z_26 += features[34] * 0.132409f;
    z_26 += features[35] * 0.264148f;
    z_26 += features[37] * 0.225451f;
    z_26 += features[40] * 0.143321f;
    z_26 += features[41] * -0.116705f;
    z_26 += features[43] * 0.198098f;
    z_26 += features[47] * -0.134801f;
    z_26 += features[48] * 0.230203f;
    z_26 += features[50] * -0.201058f;
    z_26 += features[51] * 0.210514f;
    z_26 += features[53] * -0.181608f;
    z_26 += features[56] * -0.121142f;
    z_26 += features[59] * 0.158543f;
    z_26 += features[61] * 0.198713f;
    z_26 += features[62] * -0.154792f;
    float out_26 = 0.241436f * z_26;
    {
        float arg_x = -0.048108f * z_26 + -0.048549f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.039740f * z_26 + 0.048483f) + 1e-6f;
        out_26 += -0.058892f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.157439f * z_26 + -0.026901f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.168597f * z_26 + 0.046333f) + 1e-6f;
        out_26 += 0.033832f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.014532f * z_26 + -0.061532f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.043440f * z_26 + 0.055514f) + 1e-6f;
        out_26 += -0.048709f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.138836f * z_26 + -0.034125f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.129750f * z_26 + 0.048930f) + 1e-6f;
        out_26 += -0.001811f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[26] = out_26;

    // Node 27
    float z_27 = 0.0f;
    z_27 += features[2] * -0.140475f;
    z_27 += features[4] * 0.190782f;
    z_27 += features[6] * 0.196272f;
    z_27 += features[7] * -0.144252f;
    z_27 += features[9] * -0.149184f;
    z_27 += features[11] * 0.240934f;
    z_27 += features[15] * 0.195364f;
    z_27 += features[17] * 0.178885f;
    z_27 += features[18] * 0.162948f;
    z_27 += features[24] * 0.235453f;
    z_27 += features[26] * -0.159950f;
    z_27 += features[27] * -0.219384f;
    z_27 += features[35] * -0.222703f;
    z_27 += features[42] * 0.132869f;
    z_27 += features[43] * -0.135022f;
    z_27 += features[45] * 0.153064f;
    z_27 += features[49] * -0.212366f;
    z_27 += features[50] * 0.148812f;
    z_27 += features[52] * 0.119229f;
    z_27 += features[54] * 0.217063f;
    z_27 += features[55] * -0.128633f;
    z_27 += features[56] * -0.253527f;
    z_27 += features[58] * -0.131098f;
    z_27 += features[59] * -0.228454f;
    z_27 += features[60] * 0.176610f;
    float out_27 = 0.209092f * z_27;
    {
        float arg_x = -0.122943f * z_27 + 0.025882f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.100326f * z_27 + -0.006323f) + 1e-6f;
        out_27 += -0.054082f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.030007f * z_27 + -0.113043f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.058044f * z_27 + 0.115476f) + 1e-6f;
        out_27 += 0.018992f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.125075f * z_27 + 0.026455f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.094195f * z_27 + -0.005554f) + 1e-6f;
        out_27 += -0.053082f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.113934f * z_27 + 0.019740f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.119492f * z_27 + -0.004404f) + 1e-6f;
        out_27 += -0.046879f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[27] = out_27;

    // Node 28
    float z_28 = 0.0f;
    z_28 += features[3] * -0.233861f;
    z_28 += features[4] * 0.119514f;
    z_28 += features[11] * 0.177046f;
    z_28 += features[12] * -0.239737f;
    z_28 += features[14] * -0.296110f;
    z_28 += features[15] * 0.219032f;
    z_28 += features[16] * -0.114683f;
    z_28 += features[18] * 0.130447f;
    z_28 += features[21] * 0.292824f;
    z_28 += features[27] * -0.250860f;
    z_28 += features[29] * -0.199077f;
    z_28 += features[30] * 0.228747f;
    z_28 += features[31] * -0.181146f;
    z_28 += features[32] * -0.227946f;
    z_28 += features[34] * -0.243963f;
    z_28 += features[35] * -0.195050f;
    z_28 += features[36] * -0.110201f;
    z_28 += features[38] * -0.137606f;
    z_28 += features[39] * -0.111248f;
    z_28 += features[40] * -0.216227f;
    z_28 += features[41] * 0.171009f;
    z_28 += features[44] * -0.185061f;
    z_28 += features[45] * 0.296947f;
    z_28 += features[46] * 0.170979f;
    z_28 += features[48] * -0.282525f;
    z_28 += features[50] * 0.181051f;
    z_28 += features[54] * 0.229466f;
    z_28 += features[55] * 0.161244f;
    z_28 += features[57] * 0.121275f;
    z_28 += features[60] * 0.196200f;
    z_28 += features[62] * 0.226312f;
    z_28 += features[63] * -0.153080f;
    float out_28 = 0.258822f * z_28;
    {
        float arg_x = 0.213032f * z_28 + -0.018384f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.178678f * z_28 + 0.048840f) + 1e-6f;
        out_28 += 0.068938f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.079862f * z_28 + -0.111263f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.071402f * z_28 + 0.108412f) + 1e-6f;
        out_28 += 0.028221f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.132102f * z_28 + 0.056414f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.167662f * z_28 + -0.047134f) + 1e-6f;
        out_28 += -0.071079f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.158401f * z_28 + -0.096869f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.139455f * z_28 + 0.103143f) + 1e-6f;
        out_28 += 0.033600f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[28] = out_28;

    // Node 29
    float z_29 = 0.0f;
    z_29 += features[1] * 0.165923f;
    z_29 += features[2] * -0.173911f;
    z_29 += features[3] * 0.213353f;
    z_29 += features[5] * 0.212370f;
    z_29 += features[7] * -0.125432f;
    z_29 += features[8] * 0.136648f;
    z_29 += features[10] * 0.197536f;
    z_29 += features[11] * 0.128347f;
    z_29 += features[12] * 0.170564f;
    z_29 += features[16] * -0.263648f;
    z_29 += features[17] * 0.346113f;
    z_29 += features[18] * -0.188678f;
    z_29 += features[20] * 0.189957f;
    z_29 += features[22] * -0.138590f;
    z_29 += features[23] * 0.245885f;
    z_29 += features[24] * 0.119972f;
    z_29 += features[25] * 0.257903f;
    z_29 += features[26] * 0.228196f;
    z_29 += features[27] * -0.227794f;
    z_29 += features[29] * -0.257356f;
    z_29 += features[30] * -0.184627f;
    z_29 += features[31] * -0.203333f;
    z_29 += features[33] * 0.227735f;
    z_29 += features[35] * -0.219795f;
    z_29 += features[36] * 0.149723f;
    z_29 += features[38] * -0.308453f;
    z_29 += features[39] * -0.126445f;
    z_29 += features[41] * -0.145723f;
    z_29 += features[42] * 0.328395f;
    z_29 += features[43] * -0.155292f;
    z_29 += features[45] * 0.208499f;
    z_29 += features[47] * -0.147793f;
    z_29 += features[48] * 0.131791f;
    z_29 += features[49] * 0.188198f;
    z_29 += features[50] * 0.156970f;
    z_29 += features[53] * 0.142286f;
    z_29 += features[54] * 0.133955f;
    z_29 += features[56] * -0.202885f;
    z_29 += features[57] * -0.187168f;
    z_29 += features[58] * -0.264525f;
    z_29 += features[59] * 0.150034f;
    z_29 += features[60] * -0.261129f;
    z_29 += features[61] * 0.135050f;
    z_29 += features[62] * -0.142868f;
    float out_29 = 0.232094f * z_29;
    {
        float arg_x = 0.321418f * z_29 + 0.125637f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.178228f * z_29 + -0.055779f) + 1e-6f;
        out_29 += 0.125239f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.397868f * z_29 + 0.165279f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.187958f * z_29 + -0.059461f) + 1e-6f;
        out_29 += 0.138984f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.164465f * z_29 + -0.030458f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.153700f * z_29 + 0.096282f) + 1e-6f;
        out_29 += -0.023713f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.161205f * z_29 + -0.049597f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.170011f * z_29 + 0.110376f) + 1e-6f;
        out_29 += -0.015138f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[29] = out_29;

    // Node 30
    float z_30 = 0.0f;
    z_30 += features[1] * -0.136531f;
    z_30 += features[4] * 0.193385f;
    z_30 += features[5] * -0.204699f;
    z_30 += features[8] * 0.121526f;
    z_30 += features[10] * -0.234041f;
    z_30 += features[11] * -0.150070f;
    z_30 += features[12] * -0.214614f;
    z_30 += features[13] * -0.196828f;
    z_30 += features[15] * 0.198218f;
    z_30 += features[16] * -0.125266f;
    z_30 += features[19] * -0.115631f;
    z_30 += features[22] * 0.271395f;
    z_30 += features[25] * -0.120726f;
    z_30 += features[27] * 0.259602f;
    z_30 += features[29] * -0.157407f;
    z_30 += features[31] * -0.132195f;
    z_30 += features[32] * -0.201983f;
    z_30 += features[33] * -0.124800f;
    z_30 += features[35] * -0.148083f;
    z_30 += features[37] * -0.230868f;
    z_30 += features[39] * -0.109725f;
    z_30 += features[40] * 0.119914f;
    z_30 += features[41] * 0.261978f;
    z_30 += features[43] * -0.124370f;
    z_30 += features[44] * -0.206352f;
    z_30 += features[46] * 0.147534f;
    z_30 += features[48] * -0.175008f;
    z_30 += features[51] * -0.123417f;
    z_30 += features[53] * 0.225922f;
    z_30 += features[54] * 0.166055f;
    z_30 += features[57] * 0.214709f;
    z_30 += features[60] * 0.158541f;
    z_30 += features[61] * -0.234833f;
    z_30 += features[63] * 0.295568f;
    float out_30 = 0.230944f * z_30;
    {
        float arg_x = -0.159491f * z_30 + 0.006510f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.159640f * z_30 + 0.005263f) + 1e-6f;
        out_30 += -0.056086f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.140382f * z_30 + -0.002801f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.149377f * z_30 + 0.008275f) + 1e-6f;
        out_30 += -0.043525f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.149483f * z_30 + -0.095574f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.133769f * z_30 + 0.105913f) + 1e-6f;
        out_30 += 0.001967f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.137979f * z_30 + -0.006207f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.172741f * z_30 + 0.008432f) + 1e-6f;
        out_30 += -0.038039f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[30] = out_30;

    // Node 31
    float z_31 = 0.0f;
    z_31 += features[0] * 0.257342f;
    z_31 += features[3] * 0.208569f;
    z_31 += features[4] * 0.127212f;
    z_31 += features[5] * 0.139052f;
    z_31 += features[6] * 0.124766f;
    z_31 += features[9] * 0.174923f;
    z_31 += features[10] * 0.241871f;
    z_31 += features[11] * 0.293211f;
    z_31 += features[13] * 0.181046f;
    z_31 += features[14] * -0.113521f;
    z_31 += features[17] * 0.133750f;
    z_31 += features[18] * 0.137824f;
    z_31 += features[19] * -0.182372f;
    z_31 += features[21] * 0.169036f;
    z_31 += features[22] * -0.223582f;
    z_31 += features[23] * 0.206461f;
    z_31 += features[26] * 0.130805f;
    z_31 += features[27] * -0.125455f;
    z_31 += features[29] * -0.231116f;
    z_31 += features[31] * -0.146297f;
    z_31 += features[33] * 0.186256f;
    z_31 += features[34] * 0.140922f;
    z_31 += features[35] * -0.200869f;
    z_31 += features[36] * -0.243166f;
    z_31 += features[37] * -0.189223f;
    z_31 += features[38] * -0.157778f;
    z_31 += features[41] * -0.244125f;
    z_31 += features[42] * -0.211556f;
    z_31 += features[43] * 0.153839f;
    z_31 += features[45] * 0.183685f;
    z_31 += features[46] * 0.233059f;
    z_31 += features[47] * 0.186340f;
    z_31 += features[48] * -0.250943f;
    z_31 += features[49] * 0.163408f;
    z_31 += features[50] * -0.142568f;
    z_31 += features[53] * 0.239452f;
    z_31 += features[54] * -0.181341f;
    z_31 += features[56] * 0.243278f;
    float out_31 = 0.236369f * z_31;
    {
        float arg_x = -0.138885f * z_31 + -0.010200f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.138398f * z_31 + 0.013017f) + 1e-6f;
        out_31 += -0.041299f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.064884f * z_31 + -0.057438f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.121539f * z_31 + 0.048400f) + 1e-6f;
        out_31 += 0.024815f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.103090f * z_31 + -0.052425f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.116489f * z_31 + 0.060483f) + 1e-6f;
        out_31 += 0.027768f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.158499f * z_31 + 0.000232f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.152818f * z_31 + 0.009311f) + 1e-6f;
        out_31 += -0.050628f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[31] = out_31;

    // Node 32
    float z_32 = 0.0f;
    z_32 += features[0] * 0.157777f;
    z_32 += features[1] * -0.180688f;
    z_32 += features[5] * -0.150295f;
    z_32 += features[6] * 0.194508f;
    z_32 += features[7] * 0.139930f;
    z_32 += features[8] * 0.219267f;
    z_32 += features[10] * -0.185103f;
    z_32 += features[11] * -0.162553f;
    z_32 += features[12] * 0.113847f;
    z_32 += features[13] * -0.169325f;
    z_32 += features[20] * -0.164511f;
    z_32 += features[21] * 0.190993f;
    z_32 += features[23] * -0.199055f;
    z_32 += features[28] * 0.183712f;
    z_32 += features[29] * 0.244293f;
    z_32 += features[30] * 0.150069f;
    z_32 += features[33] * -0.110690f;
    z_32 += features[34] * -0.138263f;
    z_32 += features[37] * -0.114872f;
    z_32 += features[41] * 0.196767f;
    z_32 += features[42] * 0.209582f;
    z_32 += features[43] * -0.138905f;
    z_32 += features[45] * -0.256666f;
    z_32 += features[46] * 0.118965f;
    z_32 += features[49] * -0.232139f;
    z_32 += features[50] * 0.167770f;
    z_32 += features[52] * 0.205004f;
    z_32 += features[57] * 0.248279f;
    z_32 += features[58] * -0.119562f;
    z_32 += features[59] * -0.204990f;
    z_32 += features[60] * 0.113461f;
    z_32 += features[62] * 0.213530f;
    float out_32 = 0.224421f * z_32;
    {
        float arg_x = -0.108903f * z_32 + -0.032966f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.110690f * z_32 + 0.034302f) + 1e-6f;
        out_32 += -0.006349f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.118147f * z_32 + -0.024745f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.135721f * z_32 + 0.027002f) + 1e-6f;
        out_32 += -0.020886f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.125617f * z_32 + -0.023026f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.131343f * z_32 + 0.027646f) + 1e-6f;
        out_32 += -0.021026f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.148570f * z_32 + -0.018686f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.126785f * z_32 + 0.029081f) + 1e-6f;
        out_32 += -0.022244f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[32] = out_32;

    // Node 33
    float z_33 = 0.0f;
    z_33 += features[1] * -0.273508f;
    z_33 += features[2] * -0.188933f;
    z_33 += features[3] * 0.154325f;
    z_33 += features[5] * -0.274142f;
    z_33 += features[7] * 0.205892f;
    z_33 += features[8] * -0.114649f;
    z_33 += features[10] * 0.196495f;
    z_33 += features[11] * -0.160517f;
    z_33 += features[12] * -0.217737f;
    z_33 += features[16] * 0.345790f;
    z_33 += features[17] * -0.178976f;
    z_33 += features[18] * 0.202262f;
    z_33 += features[24] * -0.188959f;
    z_33 += features[28] * 0.197457f;
    z_33 += features[29] * 0.244675f;
    z_33 += features[31] * -0.300075f;
    z_33 += features[33] * 0.145612f;
    z_33 += features[34] * -0.169942f;
    z_33 += features[38] * 0.238649f;
    z_33 += features[42] * -0.130473f;
    z_33 += features[44] * 0.294372f;
    z_33 += features[48] * 0.110064f;
    z_33 += features[49] * -0.227132f;
    z_33 += features[50] * 0.152224f;
    z_33 += features[53] * -0.352459f;
    z_33 += features[54] * 0.221661f;
    z_33 += features[56] * -0.137388f;
    z_33 += features[61] * 0.116750f;
    z_33 += features[62] * -0.161099f;
    z_33 += features[63] * 0.170609f;
    float out_33 = 0.232218f * z_33;
    {
        float arg_x = -0.233283f * z_33 + 0.054929f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.187854f * z_33 + -0.021214f) + 1e-6f;
        out_33 += -0.093485f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.209794f * z_33 + 0.067868f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.170633f * z_33 + -0.042555f) + 1e-6f;
        out_33 += -0.113803f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.245432f * z_33 + 0.034526f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.231730f * z_33 + 0.000574f) + 1e-6f;
        out_33 += -0.099210f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.190793f * z_33 + 0.033661f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.198598f * z_33 + -0.021021f) + 1e-6f;
        out_33 += -0.084973f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[33] = out_33;

    // Node 34
    float z_34 = 0.0f;
    z_34 += features[0] * -0.243209f;
    z_34 += features[1] * -0.110540f;
    z_34 += features[3] * -0.140273f;
    z_34 += features[5] * -0.117538f;
    z_34 += features[6] * -0.143315f;
    z_34 += features[9] * -0.138337f;
    z_34 += features[10] * 0.161553f;
    z_34 += features[11] * -0.153951f;
    z_34 += features[12] * 0.209449f;
    z_34 += features[13] * 0.126797f;
    z_34 += features[16] * 0.183806f;
    z_34 += features[19] * 0.114575f;
    z_34 += features[22] * -0.126527f;
    z_34 += features[27] * -0.226572f;
    z_34 += features[28] * 0.149465f;
    z_34 += features[29] * 0.233075f;
    z_34 += features[32] * 0.161292f;
    z_34 += features[33] * 0.166847f;
    z_34 += features[34] * -0.149494f;
    z_34 += features[35] * 0.207658f;
    z_34 += features[37] * 0.185713f;
    z_34 += features[44] * 0.219002f;
    z_34 += features[51] * 0.164880f;
    z_34 += features[53] * -0.271206f;
    z_34 += features[55] * 0.212144f;
    z_34 += features[57] * -0.186162f;
    z_34 += features[58] * 0.140328f;
    z_34 += features[61] * 0.175674f;
    z_34 += features[62] * 0.123001f;
    float out_34 = 0.196047f * z_34;
    {
        float arg_x = 0.160490f * z_34 + -0.048920f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.094185f * z_34 + 0.061757f) + 1e-6f;
        out_34 += 0.012682f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.127929f * z_34 + -0.009944f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.091603f * z_34 + 0.015230f) + 1e-6f;
        out_34 += 0.044403f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.112446f * z_34 + -0.075194f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.122214f * z_34 + 0.075441f) + 1e-6f;
        out_34 += 0.004950f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.111563f * z_34 + -0.029973f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.076360f * z_34 + 0.032088f) + 1e-6f;
        out_34 += 0.019198f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[34] = out_34;

    // Node 35
    float z_35 = 0.0f;
    z_35 += features[1] * 0.192550f;
    z_35 += features[3] * 0.138710f;
    z_35 += features[4] * -0.259433f;
    z_35 += features[6] * -0.209753f;
    z_35 += features[8] * -0.227500f;
    z_35 += features[9] * 0.125083f;
    z_35 += features[12] * 0.199020f;
    z_35 += features[13] * 0.158623f;
    z_35 += features[14] * 0.138336f;
    z_35 += features[16] * 0.148731f;
    z_35 += features[18] * -0.226198f;
    z_35 += features[20] * 0.164762f;
    z_35 += features[21] * -0.208503f;
    z_35 += features[22] * -0.224639f;
    z_35 += features[25] * 0.143431f;
    z_35 += features[29] * 0.120398f;
    z_35 += features[31] * 0.119596f;
    z_35 += features[32] * 0.150142f;
    z_35 += features[33] * 0.192701f;
    z_35 += features[34] * 0.204227f;
    z_35 += features[35] * 0.165497f;
    z_35 += features[37] * 0.142977f;
    z_35 += features[38] * 0.188854f;
    z_35 += features[41] * -0.212499f;
    z_35 += features[42] * -0.208913f;
    z_35 += features[44] * 0.235571f;
    z_35 += features[45] * -0.233887f;
    z_35 += features[49] * 0.251244f;
    z_35 += features[50] * -0.276742f;
    z_35 += features[51] * 0.151016f;
    z_35 += features[55] * 0.141612f;
    z_35 += features[56] * 0.158612f;
    z_35 += features[58] * 0.252541f;
    z_35 += features[60] * -0.184587f;
    z_35 += features[61] * -0.159119f;
    float out_35 = 0.233018f * z_35;
    {
        float arg_x = -0.115444f * z_35 + -0.056500f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.106486f * z_35 + 0.070518f) + 1e-6f;
        out_35 += -0.037687f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.152097f * z_35 + 0.033716f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.141067f * z_35 + -0.009665f) + 1e-6f;
        out_35 += 0.069847f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.136142f * z_35 + 0.025460f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.128505f * z_35 + -0.008197f) + 1e-6f;
        out_35 += 0.068578f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.076738f * z_35 + -0.092068f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.069370f * z_35 + 0.090207f) + 1e-6f;
        out_35 += -0.003365f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[35] = out_35;

    // Node 36
    float z_36 = 0.0f;
    z_36 += features[0] * -0.121414f;
    z_36 += features[5] * -0.177809f;
    z_36 += features[7] * 0.216028f;
    z_36 += features[10] * 0.136719f;
    z_36 += features[11] * -0.196752f;
    z_36 += features[12] * 0.222139f;
    z_36 += features[13] * 0.174085f;
    z_36 += features[14] * -0.150508f;
    z_36 += features[17] * -0.128167f;
    z_36 += features[20] * -0.124205f;
    z_36 += features[21] * 0.164491f;
    z_36 += features[24] * 0.168105f;
    z_36 += features[26] * -0.242314f;
    z_36 += features[28] * 0.189974f;
    z_36 += features[29] * 0.186128f;
    z_36 += features[31] * -0.224710f;
    z_36 += features[34] * -0.210192f;
    z_36 += features[35] * -0.140718f;
    z_36 += features[36] * -0.211649f;
    z_36 += features[38] * 0.133114f;
    z_36 += features[40] * -0.149511f;
    z_36 += features[42] * 0.121866f;
    z_36 += features[44] * 0.117818f;
    z_36 += features[48] * 0.196393f;
    z_36 += features[49] * -0.229464f;
    z_36 += features[51] * -0.181013f;
    z_36 += features[52] * 0.173380f;
    z_36 += features[53] * -0.118920f;
    z_36 += features[55] * 0.224804f;
    z_36 += features[56] * -0.237066f;
    z_36 += features[58] * -0.182793f;
    z_36 += features[59] * -0.235295f;
    float out_36 = 0.234810f * z_36;
    {
        float arg_x = 0.109587f * z_36 + -0.045917f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.100140f * z_36 + 0.049809f) + 1e-6f;
        out_36 += 0.024904f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.102252f * z_36 + -0.092350f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.113502f * z_36 + 0.094270f) + 1e-6f;
        out_36 += 0.004148f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.109576f * z_36 + -0.053957f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.150782f * z_36 + 0.055399f) + 1e-6f;
        out_36 += 0.020292f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.095486f * z_36 + -0.058492f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.110474f * z_36 + 0.058849f) + 1e-6f;
        out_36 += 0.014262f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[36] = out_36;

    // Node 37
    float z_37 = 0.0f;
    z_37 += features[1] * -0.188584f;
    z_37 += features[3] * -0.114473f;
    z_37 += features[7] * 0.142047f;
    z_37 += features[8] * 0.169319f;
    z_37 += features[9] * -0.203469f;
    z_37 += features[10] * -0.175383f;
    z_37 += features[12] * -0.151467f;
    z_37 += features[15] * 0.194594f;
    z_37 += features[17] * 0.116098f;
    z_37 += features[18] * 0.129015f;
    z_37 += features[20] * -0.138603f;
    z_37 += features[21] * -0.217527f;
    z_37 += features[22] * 0.239294f;
    z_37 += features[23] * -0.173033f;
    z_37 += features[25] * -0.263280f;
    z_37 += features[28] * -0.195477f;
    z_37 += features[29] * 0.178595f;
    z_37 += features[30] * 0.128611f;
    z_37 += features[33] * -0.228186f;
    z_37 += features[37] * 0.135011f;
    z_37 += features[38] * -0.185084f;
    z_37 += features[41] * 0.183517f;
    z_37 += features[43] * -0.138437f;
    z_37 += features[44] * -0.139928f;
    z_37 += features[48] * 0.210500f;
    z_37 += features[50] * 0.158052f;
    z_37 += features[54] * 0.173056f;
    z_37 += features[55] * -0.186132f;
    z_37 += features[56] * -0.185829f;
    z_37 += features[58] * -0.207842f;
    z_37 += features[59] * -0.152477f;
    z_37 += features[63] * 0.132281f;
    float out_37 = 0.208565f * z_37;
    {
        float arg_x = 0.107378f * z_37 + -0.071517f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.047845f * z_37 + 0.076804f) + 1e-6f;
        out_37 += 0.031634f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.081345f * z_37 + -0.022427f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.089875f * z_37 + 0.035002f) + 1e-6f;
        out_37 += -0.021453f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.094093f * z_37 + -0.009880f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.108423f * z_37 + 0.026052f) + 1e-6f;
        out_37 += -0.055941f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.101855f * z_37 + -0.007776f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.094965f * z_37 + 0.031476f) + 1e-6f;
        out_37 += -0.028808f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[37] = out_37;

    // Node 38
    float z_38 = 0.0f;
    z_38 += features[0] * -0.134007f;
    z_38 += features[2] * -0.176968f;
    z_38 += features[3] * 0.125313f;
    z_38 += features[5] * -0.140711f;
    z_38 += features[6] * -0.125032f;
    z_38 += features[7] * 0.122109f;
    z_38 += features[8] * -0.130447f;
    z_38 += features[9] * 0.164999f;
    z_38 += features[13] * 0.170082f;
    z_38 += features[14] * 0.202148f;
    z_38 += features[15] * -0.171148f;
    z_38 += features[17] * -0.182665f;
    z_38 += features[18] * -0.173994f;
    z_38 += features[19] * 0.120255f;
    z_38 += features[23] * 0.161117f;
    z_38 += features[24] * -0.144195f;
    z_38 += features[26] * 0.213191f;
    z_38 += features[27] * -0.196089f;
    z_38 += features[28] * 0.154290f;
    z_38 += features[32] * 0.204024f;
    z_38 += features[33] * 0.133351f;
    z_38 += features[36] * -0.121831f;
    z_38 += features[42] * -0.142502f;
    z_38 += features[43] * 0.152621f;
    z_38 += features[44] * 0.239735f;
    z_38 += features[47] * -0.131642f;
    z_38 += features[48] * 0.162788f;
    z_38 += features[49] * 0.242521f;
    z_38 += features[52] * -0.127892f;
    z_38 += features[53] * -0.133019f;
    float out_38 = 0.177361f * z_38;
    {
        float arg_x = 0.063811f * z_38 + -0.051093f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.110665f * z_38 + 0.049157f) + 1e-6f;
        out_38 += 0.001538f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.050692f * z_38 + -0.057700f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.088703f * z_38 + 0.054473f) + 1e-6f;
        out_38 += -0.004549f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.075202f * z_38 + -0.073384f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.067729f * z_38 + 0.075468f) + 1e-6f;
        out_38 += -0.013470f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.092816f * z_38 + -0.029717f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.120471f * z_38 + 0.032549f) + 1e-6f;
        out_38 += 0.032010f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[38] = out_38;

    // Node 39
    float z_39 = 0.0f;
    z_39 += features[2] * 0.130839f;
    z_39 += features[8] * -0.168465f;
    z_39 += features[9] * -0.209742f;
    z_39 += features[10] * -0.218157f;
    z_39 += features[11] * -0.158940f;
    z_39 += features[16] * -0.119059f;
    z_39 += features[18] * -0.172538f;
    z_39 += features[21] * -0.184980f;
    z_39 += features[22] * -0.130892f;
    z_39 += features[24] * 0.128125f;
    z_39 += features[27] * -0.110999f;
    z_39 += features[29] * 0.204490f;
    z_39 += features[30] * -0.241696f;
    z_39 += features[31] * 0.112178f;
    z_39 += features[32] * 0.225741f;
    z_39 += features[35] * 0.200585f;
    z_39 += features[36] * 0.180146f;
    z_39 += features[37] * 0.178389f;
    z_39 += features[38] * 0.117327f;
    z_39 += features[39] * 0.226547f;
    z_39 += features[40] * 0.193326f;
    z_39 += features[46] * -0.166904f;
    z_39 += features[47] * -0.207110f;
    z_39 += features[48] * 0.135483f;
    z_39 += features[49] * 0.170865f;
    z_39 += features[52] * -0.192469f;
    z_39 += features[53] * -0.127027f;
    z_39 += features[56] * -0.219134f;
    z_39 += features[61] * -0.133670f;
    z_39 += features[62] * 0.110806f;
    z_39 += features[63] * -0.211556f;
    float out_39 = 0.197132f * z_39;
    {
        float arg_x = 0.142000f * z_39 + -0.042800f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.089538f * z_39 + 0.042439f) + 1e-6f;
        out_39 += 0.002508f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.088514f * z_39 + -0.058574f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.132261f * z_39 + 0.050283f) + 1e-6f;
        out_39 += -0.002429f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.013350f * z_39 + -0.115108f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.007401f * z_39 + 0.109692f) + 1e-6f;
        out_39 += -0.010368f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.139751f * z_39 + -0.009106f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.105346f * z_39 + 0.010918f) + 1e-6f;
        out_39 += 0.013435f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[39] = out_39;

    // Node 40
    float z_40 = 0.0f;
    z_40 += features[2] * -0.125420f;
    z_40 += features[3] * 0.262663f;
    z_40 += features[4] * 0.180343f;
    z_40 += features[7] * -0.166348f;
    z_40 += features[8] * -0.130635f;
    z_40 += features[10] * 0.226140f;
    z_40 += features[11] * 0.226723f;
    z_40 += features[18] * 0.154523f;
    z_40 += features[19] * -0.135845f;
    z_40 += features[20] * 0.237705f;
    z_40 += features[21] * 0.313861f;
    z_40 += features[24] * -0.180210f;
    z_40 += features[25] * 0.251386f;
    z_40 += features[26] * 0.239169f;
    z_40 += features[27] * -0.286340f;
    z_40 += features[29] * -0.196319f;
    z_40 += features[31] * -0.219656f;
    z_40 += features[32] * -0.353520f;
    z_40 += features[33] * 0.289396f;
    z_40 += features[34] * -0.149174f;
    z_40 += features[35] * -0.150437f;
    z_40 += features[36] * -0.212344f;
    z_40 += features[37] * -0.218053f;
    z_40 += features[38] * -0.189140f;
    z_40 += features[40] * 0.118507f;
    z_40 += features[41] * -0.251880f;
    z_40 += features[43] * -0.202453f;
    z_40 += features[44] * 0.139988f;
    z_40 += features[45] * 0.229059f;
    z_40 += features[46] * 0.337582f;
    z_40 += features[47] * 0.232859f;
    z_40 += features[48] * -0.306584f;
    z_40 += features[49] * 0.169904f;
    z_40 += features[54] * -0.110516f;
    z_40 += features[55] * 0.206434f;
    z_40 += features[56] * 0.264918f;
    z_40 += features[59] * 0.186203f;
    z_40 += features[60] * -0.159880f;
    z_40 += features[61] * 0.223468f;
    z_40 += features[62] * -0.148164f;
    float out_40 = 0.288646f * z_40;
    {
        float arg_x = 0.232314f * z_40 + 0.098622f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.206794f * z_40 + -0.032455f) + 1e-6f;
        out_40 += 0.088320f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.447560f * z_40 + -0.039751f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.266657f * z_40 + 0.107402f) + 1e-6f;
        out_40 += -0.051071f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.369729f * z_40 + 0.042380f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.212285f * z_40 + 0.053100f) + 1e-6f;
        out_40 += -0.113751f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.218709f * z_40 + 0.075115f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.221161f * z_40 + -0.020556f) + 1e-6f;
        out_40 += 0.082078f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[40] = out_40;

    // Node 41
    float z_41 = 0.0f;
    z_41 += features[2] * -0.116778f;
    z_41 += features[4] * -0.183268f;
    z_41 += features[5] * 0.153562f;
    z_41 += features[6] * 0.127016f;
    z_41 += features[7] * 0.179938f;
    z_41 += features[10] * 0.193299f;
    z_41 += features[14] * 0.251383f;
    z_41 += features[15] * -0.220081f;
    z_41 += features[17] * -0.265683f;
    z_41 += features[20] * -0.113845f;
    z_41 += features[21] * -0.227321f;
    z_41 += features[22] * -0.193429f;
    z_41 += features[27] * 0.277151f;
    z_41 += features[28] * 0.147347f;
    z_41 += features[31] * 0.197943f;
    z_41 += features[32] * 0.253225f;
    z_41 += features[34] * 0.236875f;
    z_41 += features[38] * 0.129114f;
    z_41 += features[41] * -0.238068f;
    z_41 += features[42] * -0.179216f;
    z_41 += features[43] * 0.188917f;
    z_41 += features[44] * 0.182012f;
    z_41 += features[45] * -0.226632f;
    z_41 += features[47] * -0.258615f;
    z_41 += features[49] * 0.230313f;
    z_41 += features[50] * -0.188147f;
    z_41 += features[51] * -0.113545f;
    z_41 += features[52] * 0.119846f;
    z_41 += features[54] * -0.288091f;
    z_41 += features[55] * -0.131236f;
    z_41 += features[62] * -0.187860f;
    z_41 += features[63] * 0.204712f;
    float out_41 = 0.223821f * z_41;
    {
        float arg_x = -0.202610f * z_41 + 0.000307f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.140543f * z_41 + 0.021600f) + 1e-6f;
        out_41 += -0.045581f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.258497f * z_41 + 0.071609f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.135255f * z_41 + -0.019809f) + 1e-6f;
        out_41 += -0.082189f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.144917f * z_41 + 0.011325f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.146618f * z_41 + -0.008712f) + 1e-6f;
        out_41 += -0.046130f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.178139f * z_41 + -0.008394f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.153045f * z_41 + 0.020074f) + 1e-6f;
        out_41 += -0.042739f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[41] = out_41;

    // Node 42
    float z_42 = 0.0f;
    z_42 += features[0] * -0.197480f;
    z_42 += features[2] * -0.231884f;
    z_42 += features[3] * 0.213071f;
    z_42 += features[5] * -0.210348f;
    z_42 += features[6] * 0.118505f;
    z_42 += features[11] * -0.189257f;
    z_42 += features[13] * 0.140606f;
    z_42 += features[14] * -0.170065f;
    z_42 += features[16] * 0.208169f;
    z_42 += features[17] * -0.207079f;
    z_42 += features[18] * 0.228853f;
    z_42 += features[19] * 0.191648f;
    z_42 += features[22] * -0.235180f;
    z_42 += features[24] * -0.121729f;
    z_42 += features[26] * 0.152715f;
    z_42 += features[27] * -0.196191f;
    z_42 += features[28] * 0.132623f;
    z_42 += features[30] * -0.229272f;
    z_42 += features[33] * 0.245623f;
    z_42 += features[42] * 0.143797f;
    z_42 += features[44] * 0.238076f;
    z_42 += features[45] * 0.139826f;
    z_42 += features[46] * -0.118378f;
    z_42 += features[49] * 0.160377f;
    z_42 += features[50] * -0.145662f;
    z_42 += features[52] * 0.217420f;
    z_42 += features[55] * 0.131121f;
    z_42 += features[59] * -0.128683f;
    z_42 += features[60] * -0.199989f;
    z_42 += features[61] * 0.135747f;
    float out_42 = 0.210874f * z_42;
    {
        float arg_x = 0.050762f * z_42 + -0.077089f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.076707f * z_42 + 0.073488f) + 1e-6f;
        out_42 += -0.050041f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.067712f * z_42 + -0.066330f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.105887f * z_42 + 0.064931f) + 1e-6f;
        out_42 += -0.023929f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.091778f * z_42 + -0.058210f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.095441f * z_42 + 0.058966f) + 1e-6f;
        out_42 += -0.039284f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.056929f * z_42 + -0.066305f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.125438f * z_42 + 0.064276f) + 1e-6f;
        out_42 += -0.011625f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[42] = out_42;

    // Node 43
    float z_43 = 0.0f;
    z_43 += features[8] * -0.138805f;
    z_43 += features[12] * -0.200659f;
    z_43 += features[14] * 0.206108f;
    z_43 += features[17] * -0.146918f;
    z_43 += features[19] * -0.213692f;
    z_43 += features[20] * 0.210953f;
    z_43 += features[21] * 0.124289f;
    z_43 += features[24] * -0.164830f;
    z_43 += features[26] * -0.167080f;
    z_43 += features[27] * 0.157241f;
    z_43 += features[29] * 0.172351f;
    z_43 += features[32] * -0.132494f;
    z_43 += features[33] * 0.142542f;
    z_43 += features[34] * 0.161735f;
    z_43 += features[35] * 0.123870f;
    z_43 += features[37] * -0.129944f;
    z_43 += features[38] * 0.156489f;
    z_43 += features[40] * 0.180369f;
    z_43 += features[42] * -0.156086f;
    z_43 += features[43] * 0.142198f;
    z_43 += features[45] * -0.144693f;
    z_43 += features[51] * 0.148202f;
    z_43 += features[52] * -0.237695f;
    z_43 += features[53] * -0.143553f;
    z_43 += features[54] * -0.143641f;
    z_43 += features[55] * 0.115338f;
    z_43 += features[58] * 0.202057f;
    z_43 += features[59] * 0.242328f;
    z_43 += features[60] * -0.139549f;
    float out_43 = 0.195388f * z_43;
    {
        float arg_x = -0.017927f * z_43 + -0.134830f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.001612f * z_43 + 0.137728f) + 1e-6f;
        out_43 += -0.019307f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.097615f * z_43 + -0.052422f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.122371f * z_43 + 0.047883f) + 1e-6f;
        out_43 += 0.005169f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.120977f * z_43 + -0.057513f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.103455f * z_43 + 0.059121f) + 1e-6f;
        out_43 += 0.001795f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.076531f * z_43 + -0.097240f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.071878f * z_43 + 0.093552f) + 1e-6f;
        out_43 += 0.001483f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[43] = out_43;

    // Node 44
    float z_44 = 0.0f;
    z_44 += features[1] * 0.226166f;
    z_44 += features[3] * 0.110527f;
    z_44 += features[4] * -0.248157f;
    z_44 += features[9] * 0.129985f;
    z_44 += features[10] * 0.160683f;
    z_44 += features[12] * 0.267198f;
    z_44 += features[15] * -0.265420f;
    z_44 += features[17] * -0.142128f;
    z_44 += features[18] * -0.189959f;
    z_44 += features[19] * 0.141352f;
    z_44 += features[20] * -0.152961f;
    z_44 += features[21] * -0.291613f;
    z_44 += features[23] * 0.224287f;
    z_44 += features[24] * 0.165758f;
    z_44 += features[27] * -0.122142f;
    z_44 += features[29] * 0.200129f;
    z_44 += features[31] * -0.120218f;
    z_44 += features[32] * 0.221951f;
    z_44 += features[37] * 0.144379f;
    z_44 += features[38] * 0.181482f;
    z_44 += features[41] * -0.118909f;
    z_44 += features[42] * 0.219202f;
    z_44 += features[43] * -0.118873f;
    z_44 += features[47] * -0.259893f;
    z_44 += features[48] * 0.211868f;
    z_44 += features[50] * -0.115056f;
    z_44 += features[53] * -0.188350f;
    z_44 += features[58] * -0.165167f;
    z_44 += features[60] * -0.245296f;
    z_44 += features[62] * -0.163267f;
    float out_44 = 0.189677f * z_44;
    {
        float arg_x = 0.123563f * z_44 + -0.050726f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.090853f * z_44 + 0.072524f) + 1e-6f;
        out_44 += 0.021976f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.060056f * z_44 + -0.039300f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.084088f * z_44 + 0.037274f) + 1e-6f;
        out_44 += 0.023549f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.045995f * z_44 + -0.091738f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.088154f * z_44 + 0.086209f) + 1e-6f;
        out_44 += -0.022017f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.080220f * z_44 + -0.053543f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.053822f * z_44 + 0.061369f) + 1e-6f;
        out_44 += 0.012678f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[44] = out_44;

    // Node 45
    float z_45 = 0.0f;
    z_45 += features[1] * -0.279098f;
    z_45 += features[2] * 0.129693f;
    z_45 += features[3] * -0.122414f;
    z_45 += features[4] * -0.152508f;
    z_45 += features[7] * 0.166069f;
    z_45 += features[9] * -0.208795f;
    z_45 += features[11] * -0.272106f;
    z_45 += features[14] * -0.163154f;
    z_45 += features[16] * 0.137926f;
    z_45 += features[17] * -0.267720f;
    z_45 += features[18] * -0.166173f;
    z_45 += features[19] * -0.189471f;
    z_45 += features[20] * -0.157197f;
    z_45 += features[21] * 0.162519f;
    z_45 += features[22] * 0.202817f;
    z_45 += features[23] * -0.228334f;
    z_45 += features[25] * -0.250070f;
    z_45 += features[27] * 0.166663f;
    z_45 += features[28] * -0.158206f;
    z_45 += features[29] * 0.256702f;
    z_45 += features[30] * 0.126538f;
    z_45 += features[31] * 0.216023f;
    z_45 += features[33] * -0.160771f;
    z_45 += features[34] * 0.130104f;
    z_45 += features[38] * 0.170380f;
    z_45 += features[39] * 0.245117f;
    z_45 += features[40] * -0.122185f;
    z_45 += features[41] * 0.205621f;
    z_45 += features[44] * -0.134073f;
    z_45 += features[45] * -0.212424f;
    z_45 += features[47] * 0.199265f;
    z_45 += features[48] * 0.199876f;
    z_45 += features[49] * -0.237498f;
    z_45 += features[55] * 0.129222f;
    z_45 += features[57] * 0.219832f;
    z_45 += features[58] * -0.229962f;
    z_45 += features[60] * 0.127243f;
    z_45 += features[62] * 0.115663f;
    z_45 += features[63] * -0.117141f;
    float out_45 = 0.234601f * z_45;
    {
        float arg_x = 0.126263f * z_45 + 0.024110f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.139177f * z_45 + -0.011756f) + 1e-6f;
        out_45 += 0.053777f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.088609f * z_45 + -0.092477f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.095660f * z_45 + 0.108665f) + 1e-6f;
        out_45 += -0.004944f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.102277f * z_45 + 0.009666f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.139234f * z_45 + -0.010959f) + 1e-6f;
        out_45 += 0.049073f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.154158f * z_45 + -0.025442f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.139394f * z_45 + 0.059585f) + 1e-6f;
        out_45 += -0.031987f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[45] = out_45;

    // Node 46
    float z_46 = 0.0f;
    z_46 += features[5] * 0.177891f;
    z_46 += features[6] * 0.160373f;
    z_46 += features[14] * -0.111827f;
    z_46 += features[15] * 0.179051f;
    z_46 += features[17] * 0.202055f;
    z_46 += features[20] * -0.180297f;
    z_46 += features[21] * 0.202790f;
    z_46 += features[22] * 0.236481f;
    z_46 += features[25] * -0.174416f;
    z_46 += features[28] * -0.179470f;
    z_46 += features[30] * 0.115042f;
    z_46 += features[33] * -0.237356f;
    z_46 += features[35] * -0.169757f;
    z_46 += features[36] * 0.112123f;
    z_46 += features[38] * -0.221328f;
    z_46 += features[40] * -0.174176f;
    z_46 += features[42] * 0.194810f;
    z_46 += features[46] * 0.179726f;
    z_46 += features[47] * 0.175412f;
    z_46 += features[48] * -0.176666f;
    z_46 += features[49] * -0.116144f;
    z_46 += features[52] * 0.164538f;
    z_46 += features[55] * -0.174238f;
    z_46 += features[56] * -0.187964f;
    z_46 += features[57] * 0.231891f;
    z_46 += features[58] * -0.208417f;
    z_46 += features[59] * -0.164311f;
    z_46 += features[60] * 0.147557f;
    z_46 += features[62] * 0.135695f;
    float out_46 = 0.211573f * z_46;
    {
        float arg_x = -0.097957f * z_46 + -0.020645f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.079314f * z_46 + 0.030411f) + 1e-6f;
        out_46 += -0.023612f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.104649f * z_46 + 0.006334f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.118560f * z_46 + 0.003037f) + 1e-6f;
        out_46 += -0.039692f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.055041f * z_46 + -0.138324f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.051203f * z_46 + 0.138283f) + 1e-6f;
        out_46 += 0.010936f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.051461f * z_46 + -0.077103f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.091225f * z_46 + 0.069223f) + 1e-6f;
        out_46 += -0.008259f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[46] = out_46;

    // Node 47
    float z_47 = 0.0f;
    z_47 += features[2] * -0.205815f;
    z_47 += features[6] * 0.114583f;
    z_47 += features[7] * -0.118973f;
    z_47 += features[8] * -0.113303f;
    z_47 += features[9] * 0.113138f;
    z_47 += features[11] * 0.192099f;
    z_47 += features[12] * 0.115095f;
    z_47 += features[16] * -0.144489f;
    z_47 += features[22] * -0.142633f;
    z_47 += features[23] * 0.128780f;
    z_47 += features[25] * 0.147015f;
    z_47 += features[27] * -0.183371f;
    z_47 += features[32] * 0.145300f;
    z_47 += features[35] * -0.179474f;
    z_47 += features[36] * -0.176192f;
    z_47 += features[37] * 0.120073f;
    z_47 += features[39] * -0.145057f;
    z_47 += features[45] * 0.213352f;
    z_47 += features[49] * 0.208787f;
    z_47 += features[52] * 0.131285f;
    z_47 += features[54] * 0.189869f;
    z_47 += features[55] * 0.152758f;
    z_47 += features[56] * -0.212024f;
    z_47 += features[58] * -0.194158f;
    z_47 += features[62] * -0.120795f;
    z_47 += features[63] * -0.137648f;
    float out_47 = 0.136798f * z_47;
    {
        float arg_x = 0.018291f * z_47 + -0.047230f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.025052f * z_47 + 0.045306f) + 1e-6f;
        out_47 += 0.006101f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.021630f * z_47 + -0.037092f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.072054f * z_47 + 0.033801f) + 1e-6f;
        out_47 += 0.025786f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.026446f * z_47 + -0.040771f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.016656f * z_47 + 0.038638f) + 1e-6f;
        out_47 += 0.016039f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.062540f * z_47 + -0.057175f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.026478f * z_47 + 0.056764f) + 1e-6f;
        out_47 += -0.002469f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[47] = out_47;

    // Node 48
    float z_48 = 0.0f;
    z_48 += features[3] * -0.272315f;
    z_48 += features[6] * 0.129079f;
    z_48 += features[7] * 0.152967f;
    z_48 += features[10] * 0.158785f;
    z_48 += features[14] * 0.152152f;
    z_48 += features[21] * -0.174311f;
    z_48 += features[22] * -0.160534f;
    z_48 += features[23] * 0.178354f;
    z_48 += features[25] * -0.139307f;
    z_48 += features[26] * -0.175838f;
    z_48 += features[30] * -0.143507f;
    z_48 += features[31] * 0.174722f;
    z_48 += features[32] * 0.287558f;
    z_48 += features[33] * -0.142594f;
    z_48 += features[35] * 0.189492f;
    z_48 += features[36] * 0.190288f;
    z_48 += features[41] * -0.139547f;
    z_48 += features[43] * 0.200983f;
    z_48 += features[45] * -0.190728f;
    z_48 += features[46] * -0.277549f;
    z_48 += features[48] * 0.218715f;
    z_48 += features[51] * 0.175925f;
    z_48 += features[53] * -0.207111f;
    z_48 += features[54] * 0.156410f;
    z_48 += features[56] * -0.260934f;
    z_48 += features[57] * -0.138856f;
    z_48 += features[60] * 0.153254f;
    z_48 += features[63] * 0.177330f;
    float out_48 = 0.205242f * z_48;
    {
        float arg_x = 0.162033f * z_48 + 0.008605f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.126250f * z_48 + 0.000254f) + 1e-6f;
        out_48 += 0.054871f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.023316f * z_48 + -0.105543f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.010069f * z_48 + 0.108135f) + 1e-6f;
        out_48 += -0.013660f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.071853f * z_48 + -0.132839f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.093923f * z_48 + 0.131152f) + 1e-6f;
        out_48 += -0.006567f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.070775f * z_48 + -0.052874f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.041107f * z_48 + 0.055880f) + 1e-6f;
        out_48 += -0.030723f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[48] = out_48;

    // Node 49
    float z_49 = 0.0f;
    z_49 += features[1] * -0.241842f;
    z_49 += features[2] * 0.184450f;
    z_49 += features[6] * -0.192007f;
    z_49 += features[7] * 0.137050f;
    z_49 += features[8] * -0.178522f;
    z_49 += features[10] * 0.225827f;
    z_49 += features[12] * -0.150666f;
    z_49 += features[13] * 0.177841f;
    z_49 += features[14] * 0.129969f;
    z_49 += features[17] * -0.272015f;
    z_49 += features[18] * -0.123667f;
    z_49 += features[19] * -0.231236f;
    z_49 += features[21] * 0.221855f;
    z_49 += features[23] * -0.167763f;
    z_49 += features[25] * 0.179770f;
    z_49 += features[26] * -0.270907f;
    z_49 += features[27] * 0.251722f;
    z_49 += features[30] * -0.184972f;
    z_49 += features[31] * 0.124477f;
    z_49 += features[34] * 0.235642f;
    z_49 += features[36] * -0.111322f;
    z_49 += features[38] * 0.217457f;
    z_49 += features[41] * -0.225170f;
    z_49 += features[42] * -0.258932f;
    z_49 += features[44] * 0.215518f;
    z_49 += features[45] * -0.200828f;
    z_49 += features[47] * 0.134585f;
    z_49 += features[49] * -0.114190f;
    z_49 += features[51] * 0.117327f;
    z_49 += features[55] * 0.252471f;
    z_49 += features[62] * -0.252744f;
    float out_49 = 0.215259f * z_49;
    {
        float arg_x = 0.149166f * z_49 + 0.010925f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.141885f * z_49 + 0.003604f) + 1e-6f;
        out_49 += 0.056776f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.062074f * z_49 + -0.083309f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.066475f * z_49 + 0.106662f) + 1e-6f;
        out_49 += -0.034864f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.170048f * z_49 + 0.018543f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.156668f * z_49 + 0.010058f) + 1e-6f;
        out_49 += 0.017898f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.148867f * z_49 + -0.090046f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.089869f * z_49 + 0.092001f) + 1e-6f;
        out_49 += -0.011916f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[49] = out_49;

    // Node 50
    float z_50 = 0.0f;
    z_50 += features[1] * 0.227377f;
    z_50 += features[2] * 0.172604f;
    z_50 += features[6] * -0.129845f;
    z_50 += features[8] * -0.169135f;
    z_50 += features[10] * -0.120226f;
    z_50 += features[11] * 0.221478f;
    z_50 += features[12] * -0.165894f;
    z_50 += features[16] * -0.205136f;
    z_50 += features[17] * 0.169358f;
    z_50 += features[20] * 0.237333f;
    z_50 += features[22] * 0.159408f;
    z_50 += features[23] * -0.134866f;
    z_50 += features[31] * 0.211620f;
    z_50 += features[32] * -0.120895f;
    z_50 += features[34] * 0.177028f;
    z_50 += features[35] * 0.119540f;
    z_50 += features[38] * -0.121629f;
    z_50 += features[40] * 0.193456f;
    z_50 += features[45] * -0.131448f;
    z_50 += features[49] * 0.182293f;
    z_50 += features[52] * -0.199588f;
    z_50 += features[53] * 0.267850f;
    z_50 += features[54] * -0.149048f;
    z_50 += features[60] * -0.139683f;
    z_50 += features[61] * -0.186835f;
    z_50 += features[62] * 0.144857f;
    float out_50 = 0.178975f * z_50;
    {
        float arg_x = -0.070136f * z_50 + -0.008815f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.058095f * z_50 + 0.009779f) + 1e-6f;
        out_50 += -0.041793f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.061610f * z_50 + -0.010884f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.065870f * z_50 + 0.009564f) + 1e-6f;
        out_50 += -0.040800f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.057048f * z_50 + -0.044399f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.110888f * z_50 + 0.040101f) + 1e-6f;
        out_50 += -0.010392f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.056464f * z_50 + -0.027303f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.087606f * z_50 + 0.025428f) + 1e-6f;
        out_50 += -0.015222f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[50] = out_50;

    // Node 51
    float z_51 = 0.0f;
    z_51 += features[4] * 0.141711f;
    z_51 += features[6] * -0.179609f;
    z_51 += features[10] * -0.147090f;
    z_51 += features[11] * 0.200507f;
    z_51 += features[14] * -0.138201f;
    z_51 += features[17] * 0.191925f;
    z_51 += features[20] * 0.169666f;
    z_51 += features[24] * 0.151203f;
    z_51 += features[26] * -0.114417f;
    z_51 += features[27] * -0.153743f;
    z_51 += features[28] * -0.115897f;
    z_51 += features[30] * -0.125234f;
    z_51 += features[32] * -0.218385f;
    z_51 += features[36] * 0.178094f;
    z_51 += features[38] * -0.253081f;
    z_51 += features[39] * -0.176136f;
    z_51 += features[42] * 0.261439f;
    z_51 += features[47] * 0.121064f;
    z_51 += features[52] * -0.117954f;
    z_51 += features[53] * 0.227772f;
    z_51 += features[56] * 0.155503f;
    z_51 += features[58] * -0.154496f;
    z_51 += features[59] * 0.196852f;
    z_51 += features[60] * -0.110375f;
    z_51 += features[61] * 0.156459f;
    z_51 += features[62] * 0.186562f;
    float out_51 = 0.164915f * z_51;
    {
        float arg_x = -0.075372f * z_51 + -0.077453f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.081060f * z_51 + 0.080327f) + 1e-6f;
        out_51 += 0.010076f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.014057f * z_51 + -0.024665f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.029798f * z_51 + 0.022002f) + 1e-6f;
        out_51 += 0.051634f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.113483f * z_51 + 0.037166f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.035272f * z_51 + -0.031951f) + 1e-6f;
        out_51 += 0.074992f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.109873f * z_51 + -0.041579f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.044061f * z_51 + 0.053139f) + 1e-6f;
        out_51 += 0.035295f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[51] = out_51;

    // Node 52
    float z_52 = 0.0f;
    z_52 += features[1] * -0.146235f;
    z_52 += features[2] * -0.128255f;
    z_52 += features[4] * -0.180777f;
    z_52 += features[6] * -0.142356f;
    z_52 += features[7] * 0.140009f;
    z_52 += features[9] * 0.178302f;
    z_52 += features[10] * 0.153393f;
    z_52 += features[13] * 0.198866f;
    z_52 += features[16] * 0.180459f;
    z_52 += features[17] * -0.128508f;
    z_52 += features[19] * -0.115024f;
    z_52 += features[20] * 0.127736f;
    z_52 += features[23] * -0.156118f;
    z_52 += features[26] * -0.124388f;
    z_52 += features[27] * 0.207842f;
    z_52 += features[31] * 0.180586f;
    z_52 += features[32] * -0.166519f;
    z_52 += features[33] * 0.128261f;
    z_52 += features[34] * 0.154223f;
    z_52 += features[37] * -0.180785f;
    z_52 += features[44] * 0.168897f;
    z_52 += features[45] * -0.152143f;
    z_52 += features[46] * 0.160507f;
    z_52 += features[50] * -0.116049f;
    z_52 += features[51] * -0.172305f;
    z_52 += features[54] * -0.176918f;
    z_52 += features[56] * 0.177437f;
    z_52 += features[57] * 0.130286f;
    z_52 += features[58] * 0.260790f;
    z_52 += features[61] * -0.150448f;
    z_52 += features[62] * -0.162897f;
    z_52 += features[63] * 0.195646f;
    float out_52 = 0.196913f * z_52;
    {
        float arg_x = -0.162903f * z_52 + -0.179859f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.165494f * z_52 + 0.190351f) + 1e-6f;
        out_52 += 0.000561f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.036983f * z_52 + -0.082641f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.044423f * z_52 + 0.083678f) + 1e-6f;
        out_52 += 0.009986f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.049652f * z_52 + -0.061220f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.050853f * z_52 + 0.061901f) + 1e-6f;
        out_52 += 0.014805f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.066883f * z_52 + 0.007309f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.047484f * z_52 + -0.006188f) + 1e-6f;
        out_52 += 0.037829f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[52] = out_52;

    // Node 53
    float z_53 = 0.0f;
    z_53 += features[0] * -0.123345f;
    z_53 += features[1] * 0.241772f;
    z_53 += features[2] * 0.197498f;
    z_53 += features[7] * -0.283249f;
    z_53 += features[8] * 0.192872f;
    z_53 += features[11] * -0.138709f;
    z_53 += features[12] * -0.285243f;
    z_53 += features[13] * -0.151776f;
    z_53 += features[14] * 0.333229f;
    z_53 += features[16] * -0.248408f;
    z_53 += features[17] * 0.237178f;
    z_53 += features[21] * -0.129229f;
    z_53 += features[22] * 0.210966f;
    z_53 += features[23] * -0.122235f;
    z_53 += features[24] * -0.191003f;
    z_53 += features[26] * 0.130172f;
    z_53 += features[27] * 0.289348f;
    z_53 += features[29] * -0.202746f;
    z_53 += features[30] * 0.215100f;
    z_53 += features[33] * 0.156078f;
    z_53 += features[34] * 0.132315f;
    z_53 += features[35] * 0.329592f;
    z_53 += features[36] * 0.156982f;
    z_53 += features[40] * 0.261430f;
    z_53 += features[47] * -0.149706f;
    z_53 += features[49] * -0.297953f;
    z_53 += features[50] * 0.176960f;
    z_53 += features[52] * -0.198017f;
    z_53 += features[55] * -0.218568f;
    z_53 += features[58] * 0.190080f;
    z_53 += features[61] * -0.246759f;
    z_53 += features[63] * 0.148068f;
    float out_53 = 0.276321f * z_53;
    {
        float arg_x = -0.132163f * z_53 + -0.020938f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.179448f * z_53 + 0.033126f) + 1e-6f;
        out_53 += -0.041806f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.131520f * z_53 + -0.011017f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.181063f * z_53 + 0.020409f) + 1e-6f;
        out_53 += -0.059093f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.137096f * z_53 + -0.006018f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.194007f * z_53 + 0.016883f) + 1e-6f;
        out_53 += -0.065776f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.358251f * z_53 + -0.017217f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.244266f * z_53 + 0.076610f) + 1e-6f;
        out_53 += 0.075008f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[53] = out_53;

    // Node 54
    float z_54 = 0.0f;
    z_54 += features[2] * 0.143353f;
    z_54 += features[7] * 0.182276f;
    z_54 += features[8] * -0.153910f;
    z_54 += features[10] * 0.193038f;
    z_54 += features[12] * 0.268998f;
    z_54 += features[13] * 0.178612f;
    z_54 += features[14] * 0.211087f;
    z_54 += features[16] * 0.234972f;
    z_54 += features[17] * -0.150118f;
    z_54 += features[19] * 0.145510f;
    z_54 += features[23] * -0.160371f;
    z_54 += features[25] * 0.141185f;
    z_54 += features[27] * 0.137543f;
    z_54 += features[30] * -0.198883f;
    z_54 += features[31] * 0.225869f;
    z_54 += features[32] * 0.197943f;
    z_54 += features[33] * -0.127396f;
    z_54 += features[35] * 0.140577f;
    z_54 += features[37] * 0.133823f;
    z_54 += features[38] * 0.207805f;
    z_54 += features[41] * -0.173812f;
    z_54 += features[42] * -0.275588f;
    z_54 += features[43] * 0.126245f;
    z_54 += features[44] * 0.269810f;
    z_54 += features[46] * -0.128130f;
    z_54 += features[48] * 0.164130f;
    z_54 += features[53] * -0.203155f;
    z_54 += features[54] * -0.251562f;
    z_54 += features[58] * 0.208499f;
    z_54 += features[59] * 0.202892f;
    float out_54 = 0.205345f * z_54;
    {
        float arg_x = 0.086085f * z_54 + -0.054629f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.127708f * z_54 + 0.046595f) + 1e-6f;
        out_54 += -0.017229f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.121871f * z_54 + -0.029296f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.102242f * z_54 + 0.040107f) + 1e-6f;
        out_54 += 0.015580f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.127591f * z_54 + -0.036998f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.127179f * z_54 + 0.047911f) + 1e-6f;
        out_54 += -0.008296f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.016211f * z_54 + -0.032078f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.011955f * z_54 + 0.029904f) + 1e-6f;
        out_54 += -0.047195f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[54] = out_54;

    // Node 55
    float z_55 = 0.0f;
    z_55 += features[4] * 0.173349f;
    z_55 += features[7] * -0.251795f;
    z_55 += features[8] * 0.184538f;
    z_55 += features[10] * -0.166472f;
    z_55 += features[11] * 0.157195f;
    z_55 += features[13] * -0.172794f;
    z_55 += features[15] * 0.183054f;
    z_55 += features[16] * -0.235962f;
    z_55 += features[17] * 0.220315f;
    z_55 += features[19] * -0.174329f;
    z_55 += features[23] * 0.241692f;
    z_55 += features[24] * 0.161918f;
    z_55 += features[29] * -0.252134f;
    z_55 += features[32] * -0.259347f;
    z_55 += features[38] * -0.172915f;
    z_55 += features[42] * 0.174228f;
    z_55 += features[44] * -0.258495f;
    z_55 += features[45] * 0.298004f;
    z_55 += features[47] * 0.149011f;
    z_55 += features[48] * -0.154987f;
    z_55 += features[49] * -0.198362f;
    z_55 += features[50] * 0.275329f;
    z_55 += features[51] * 0.153451f;
    z_55 += features[52] * -0.176581f;
    z_55 += features[54] * 0.127790f;
    z_55 += features[56] * -0.159914f;
    z_55 += features[57] * -0.112097f;
    z_55 += features[58] * -0.131112f;
    float out_55 = 0.243055f * z_55;
    {
        float arg_x = -0.152576f * z_55 + 0.017330f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.122504f * z_55 + 0.013589f) + 1e-6f;
        out_55 += -0.045917f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.141221f * z_55 + -0.023833f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.146404f * z_55 + 0.030756f) + 1e-6f;
        out_55 += 0.043794f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.123467f * z_55 + -0.031622f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.153320f * z_55 + 0.034131f) + 1e-6f;
        out_55 += 0.040791f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.130667f * z_55 + -0.023226f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.162653f * z_55 + 0.027368f) + 1e-6f;
        out_55 += 0.044828f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[55] = out_55;

    // Node 56
    float z_56 = 0.0f;
    z_56 += features[0] * -0.256128f;
    z_56 += features[1] * -0.156753f;
    z_56 += features[2] * -0.156335f;
    z_56 += features[4] * 0.184891f;
    z_56 += features[9] * 0.205160f;
    z_56 += features[10] * 0.149247f;
    z_56 += features[14] * 0.217725f;
    z_56 += features[18] * 0.152474f;
    z_56 += features[20] * -0.117062f;
    z_56 += features[25] * 0.244159f;
    z_56 += features[27] * 0.143582f;
    z_56 += features[33] * 0.189474f;
    z_56 += features[34] * 0.206244f;
    z_56 += features[36] * -0.139747f;
    z_56 += features[38] * 0.228102f;
    z_56 += features[39] * -0.245124f;
    z_56 += features[40] * 0.119068f;
    z_56 += features[41] * -0.224170f;
    z_56 += features[42] * -0.201887f;
    z_56 += features[44] * 0.236708f;
    z_56 += features[48] * -0.193315f;
    z_56 += features[49] * -0.121803f;
    z_56 += features[50] * 0.164708f;
    z_56 += features[51] * -0.143902f;
    z_56 += features[57] * -0.237096f;
    z_56 += features[58] * 0.260647f;
    z_56 += features[61] * -0.141309f;
    z_56 += features[62] * -0.174058f;
    z_56 += features[63] * 0.263484f;
    float out_56 = 0.230586f * z_56;
    {
        float arg_x = 0.041103f * z_56 + -0.100088f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.055838f * z_56 + 0.097841f) + 1e-6f;
        out_56 += -0.014039f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.063125f * z_56 + -0.094692f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.071311f * z_56 + 0.093998f) + 1e-6f;
        out_56 += -0.011885f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.158519f * z_56 + 0.003786f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.147659f * z_56 + 0.020956f) + 1e-6f;
        out_56 += 0.039605f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.076767f * z_56 + -0.042708f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.083405f * z_56 + 0.044043f) + 1e-6f;
        out_56 += -0.039063f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[56] = out_56;

    // Node 57
    float z_57 = 0.0f;
    z_57 += features[0] * 0.229918f;
    z_57 += features[1] * -0.114628f;
    z_57 += features[6] * 0.201461f;
    z_57 += features[7] * 0.160874f;
    z_57 += features[8] * 0.157591f;
    z_57 += features[9] * -0.187618f;
    z_57 += features[10] * -0.153342f;
    z_57 += features[12] * -0.167409f;
    z_57 += features[14] * -0.159264f;
    z_57 += features[16] * -0.174241f;
    z_57 += features[17] * 0.169781f;
    z_57 += features[19] * -0.219688f;
    z_57 += features[21] * 0.228324f;
    z_57 += features[22] * 0.166298f;
    z_57 += features[23] * -0.196329f;
    z_57 += features[25] * -0.199521f;
    z_57 += features[26] * -0.170165f;
    z_57 += features[29] * 0.117670f;
    z_57 += features[30] * 0.211771f;
    z_57 += features[33] * -0.225015f;
    z_57 += features[38] * -0.229842f;
    z_57 += features[40] * -0.144365f;
    z_57 += features[42] * 0.152582f;
    z_57 += features[43] * -0.134273f;
    z_57 += features[44] * -0.146947f;
    z_57 += features[46] * 0.198326f;
    z_57 += features[47] * 0.216315f;
    z_57 += features[48] * -0.250505f;
    z_57 += features[49] * -0.173539f;
    z_57 += features[50] * 0.146691f;
    z_57 += features[53] * 0.256616f;
    z_57 += features[56] * 0.156867f;
    z_57 += features[57] * 0.139151f;
    z_57 += features[58] * -0.228820f;
    z_57 += features[62] * 0.121404f;
    z_57 += features[63] * -0.172028f;
    float out_57 = 0.219438f * z_57;
    {
        float arg_x = -0.114079f * z_57 + -0.023127f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.119462f * z_57 + 0.037401f) + 1e-6f;
        out_57 += -0.007125f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.003950f * z_57 + -0.080656f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.009802f * z_57 + 0.077251f) + 1e-6f;
        out_57 += 0.050123f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.113987f * z_57 + -0.018623f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.143239f * z_57 + 0.030646f) + 1e-6f;
        out_57 += -0.022543f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.102035f * z_57 + -0.041391f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.091294f * z_57 + 0.049694f) + 1e-6f;
        out_57 += 0.013749f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[57] = out_57;

    // Node 58
    float z_58 = 0.0f;
    z_58 += features[0] * -0.211713f;
    z_58 += features[4] * -0.186109f;
    z_58 += features[5] * -0.123712f;
    z_58 += features[10] * 0.190002f;
    z_58 += features[12] * 0.257434f;
    z_58 += features[13] * 0.116655f;
    z_58 += features[15] * -0.266879f;
    z_58 += features[16] * 0.260379f;
    z_58 += features[17] * -0.260551f;
    z_58 += features[18] * -0.148777f;
    z_58 += features[20] * 0.136819f;
    z_58 += features[22] * -0.139900f;
    z_58 += features[25] * 0.138839f;
    z_58 += features[27] * -0.242943f;
    z_58 += features[28] * 0.157455f;
    z_58 += features[29] * 0.200178f;
    z_58 += features[30] * -0.269952f;
    z_58 += features[32] * 0.246992f;
    z_58 += features[33] * 0.144244f;
    z_58 += features[34] * -0.163969f;
    z_58 += features[35] * 0.226193f;
    z_58 += features[36] * -0.127117f;
    z_58 += features[38] * 0.274441f;
    z_58 += features[43] * 0.182542f;
    z_58 += features[44] * 0.235745f;
    z_58 += features[45] * -0.176249f;
    z_58 += features[48] * 0.117818f;
    z_58 += features[49] * 0.115003f;
    z_58 += features[50] * -0.140264f;
    z_58 += features[53] * -0.243000f;
    z_58 += features[55] * 0.171951f;
    z_58 += features[58] * 0.192999f;
    z_58 += features[59] * 0.172280f;
    z_58 += features[61] * 0.140389f;
    z_58 += features[62] * 0.161575f;
    z_58 += features[63] * -0.219658f;
    float out_58 = 0.244319f * z_58;
    {
        float arg_x = -0.127583f * z_58 + 0.034818f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.129704f * z_58 + -0.010843f) + 1e-6f;
        out_58 += -0.100137f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.089376f * z_58 + -0.153727f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.060865f * z_58 + 0.153278f) + 1e-6f;
        out_58 += -0.001428f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.080111f * z_58 + -0.025732f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.143943f * z_58 + 0.029486f) + 1e-6f;
        out_58 += -0.051308f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.219389f * z_58 + 0.058064f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.171202f * z_58 + 0.015342f) + 1e-6f;
        out_58 += 0.103866f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[58] = out_58;

    // Node 59
    float z_59 = 0.0f;
    z_59 += features[0] * -0.144615f;
    z_59 += features[6] * -0.157038f;
    z_59 += features[7] * 0.144985f;
    z_59 += features[8] * -0.194000f;
    z_59 += features[11] * -0.137896f;
    z_59 += features[12] * -0.240704f;
    z_59 += features[14] * 0.152210f;
    z_59 += features[16] * 0.200393f;
    z_59 += features[19] * -0.154960f;
    z_59 += features[20] * 0.224702f;
    z_59 += features[21] * 0.129379f;
    z_59 += features[22] * 0.193368f;
    z_59 += features[23] * -0.186030f;
    z_59 += features[25] * 0.123299f;
    z_59 += features[29] * 0.204071f;
    z_59 += features[30] * -0.203453f;
    z_59 += features[33] * -0.210568f;
    z_59 += features[34] * 0.167191f;
    z_59 += features[37] * 0.140306f;
    z_59 += features[38] * 0.132232f;
    z_59 += features[40] * -0.185389f;
    z_59 += features[42] * -0.228219f;
    z_59 += features[43] * 0.194994f;
    z_59 += features[45] * -0.123908f;
    z_59 += features[47] * 0.165935f;
    z_59 += features[52] * -0.242372f;
    z_59 += features[56] * 0.116996f;
    z_59 += features[58] * 0.201471f;
    z_59 += features[59] * 0.139580f;
    z_59 += features[60] * 0.171082f;
    z_59 += features[61] * 0.162955f;
    z_59 += features[62] * 0.151916f;
    float out_59 = 0.195301f * z_59;
    {
        float arg_x = 0.085223f * z_59 + 0.016024f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.085786f * z_59 + -0.012164f) + 1e-6f;
        out_59 += 0.060237f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.126635f * z_59 + 0.050847f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.111846f * z_59 + -0.039662f) + 1e-6f;
        out_59 += 0.113179f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.085704f * z_59 + -0.061877f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.079302f * z_59 + 0.076607f) + 1e-6f;
        out_59 += 0.012254f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.103205f * z_59 + -0.054670f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.058740f * z_59 + 0.080609f) + 1e-6f;
        out_59 += 0.011636f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[59] = out_59;

    // Node 60
    float z_60 = 0.0f;
    z_60 += features[1] * 0.247473f;
    z_60 += features[3] * 0.176068f;
    z_60 += features[6] * -0.141904f;
    z_60 += features[8] * -0.119272f;
    z_60 += features[12] * 0.159167f;
    z_60 += features[16] * -0.117154f;
    z_60 += features[19] * 0.203956f;
    z_60 += features[20] * 0.152538f;
    z_60 += features[25] * 0.119702f;
    z_60 += features[26] * 0.197403f;
    z_60 += features[27] * -0.202820f;
    z_60 += features[28] * 0.172580f;
    z_60 += features[30] * -0.207818f;
    z_60 += features[31] * 0.176092f;
    z_60 += features[33] * 0.255476f;
    z_60 += features[34] * 0.201053f;
    z_60 += features[36] * -0.134627f;
    z_60 += features[38] * 0.113848f;
    z_60 += features[41] * -0.165425f;
    z_60 += features[44] * 0.193846f;
    z_60 += features[47] * -0.180264f;
    z_60 += features[49] * 0.147146f;
    z_60 += features[50] * -0.235783f;
    z_60 += features[54] * -0.194635f;
    z_60 += features[58] * 0.188221f;
    z_60 += features[59] * 0.188669f;
    z_60 += features[60] * -0.120293f;
    float out_60 = 0.213593f * z_60;
    {
        float arg_x = 0.114943f * z_60 + 0.000017f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.118539f * z_60 + 0.012781f) + 1e-6f;
        out_60 += 0.031762f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.062543f * z_60 + -0.090996f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.086776f * z_60 + 0.087919f) + 1e-6f;
        out_60 += 0.002807f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.063803f * z_60 + -0.046204f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.108308f * z_60 + 0.040791f) + 1e-6f;
        out_60 += 0.012285f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.107986f * z_60 + 0.000122f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.087112f * z_60 + 0.010356f) + 1e-6f;
        out_60 += 0.034999f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[60] = out_60;

    // Node 61
    float z_61 = 0.0f;
    z_61 += features[1] * 0.216530f;
    z_61 += features[5] * 0.202750f;
    z_61 += features[7] * -0.164289f;
    z_61 += features[8] * 0.143819f;
    z_61 += features[10] * -0.110872f;
    z_61 += features[12] * 0.184882f;
    z_61 += features[13] * -0.147484f;
    z_61 += features[15] * 0.125818f;
    z_61 += features[16] * -0.119562f;
    z_61 += features[18] * -0.195969f;
    z_61 += features[21] * -0.186406f;
    z_61 += features[22] * 0.143266f;
    z_61 += features[24] * 0.112095f;
    z_61 += features[25] * -0.130258f;
    z_61 += features[27] * -0.274649f;
    z_61 += features[29] * 0.184551f;
    z_61 += features[31] * 0.144360f;
    z_61 += features[33] * -0.198920f;
    z_61 += features[34] * -0.172249f;
    z_61 += features[36] * 0.239615f;
    z_61 += features[38] * -0.237810f;
    z_61 += features[39] * -0.135105f;
    z_61 += features[44] * -0.224986f;
    z_61 += features[45] * 0.159202f;
    z_61 += features[48] * 0.135222f;
    z_61 += features[54] * 0.165873f;
    z_61 += features[55] * -0.176764f;
    z_61 += features[59] * -0.122675f;
    z_61 += features[60] * 0.143556f;
    z_61 += features[62] * 0.126449f;
    float out_61 = 0.230183f * z_61;
    {
        float arg_x = 0.009884f * z_61 + -0.056609f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.014303f * z_61 + 0.046870f) + 1e-6f;
        out_61 += 0.029993f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.003131f * z_61 + -0.057852f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.000255f * z_61 + 0.053308f) + 1e-6f;
        out_61 += 0.033272f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.099585f * z_61 + -0.106973f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.093090f * z_61 + 0.109985f) + 1e-6f;
        out_61 += 0.002772f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.002165f * z_61 + -0.056617f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.013617f * z_61 + 0.051604f) + 1e-6f;
        out_61 += 0.030424f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[61] = out_61;

    // Node 62
    float z_62 = 0.0f;
    z_62 += features[3] * -0.202275f;
    z_62 += features[4] * -0.153116f;
    z_62 += features[8] * 0.150671f;
    z_62 += features[10] * -0.268651f;
    z_62 += features[11] * -0.239732f;
    z_62 += features[12] * -0.164722f;
    z_62 += features[13] * -0.137934f;
    z_62 += features[18] * -0.192453f;
    z_62 += features[19] * 0.161592f;
    z_62 += features[21] * -0.167417f;
    z_62 += features[22] * 0.133821f;
    z_62 += features[23] * -0.199700f;
    z_62 += features[24] * 0.163050f;
    z_62 += features[25] * -0.220645f;
    z_62 += features[26] * -0.198913f;
    z_62 += features[27] * 0.135536f;
    z_62 += features[28] * -0.174436f;
    z_62 += features[30] * 0.199682f;
    z_62 += features[31] * 0.135449f;
    z_62 += features[33] * -0.269368f;
    z_62 += features[40] * 0.204631f;
    z_62 += features[41] * 0.181614f;
    z_62 += features[44] * -0.223024f;
    z_62 += features[45] * -0.142381f;
    z_62 += features[47] * -0.140423f;
    z_62 += features[49] * -0.246897f;
    z_62 += features[52] * -0.130895f;
    z_62 += features[53] * 0.149418f;
    z_62 += features[58] * -0.147422f;
    z_62 += features[60] * 0.175983f;
    float out_62 = 0.196960f * z_62;
    {
        float arg_x = 0.216071f * z_62 + -0.028622f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.125024f * z_62 + 0.041902f) + 1e-6f;
        out_62 += 0.043146f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.186992f * z_62 + -0.029670f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.140846f * z_62 + 0.036994f) + 1e-6f;
        out_62 += 0.047657f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.054350f * z_62 + -0.081022f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.007188f * z_62 + 0.088605f) + 1e-6f;
        out_62 += -0.018670f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.217267f * z_62 + -0.034143f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.131253f * z_62 + 0.046552f) + 1e-6f;
        out_62 += 0.037681f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[62] = out_62;

    // Node 63
    float z_63 = 0.0f;
    z_63 += features[0] * -0.138290f;
    z_63 += features[4] * 0.210497f;
    z_63 += features[6] * 0.137433f;
    z_63 += features[10] * 0.164154f;
    z_63 += features[13] * 0.148872f;
    z_63 += features[14] * -0.155258f;
    z_63 += features[16] * 0.130775f;
    z_63 += features[17] * -0.193092f;
    z_63 += features[19] * -0.222992f;
    z_63 += features[21] * 0.229957f;
    z_63 += features[24] * -0.257096f;
    z_63 += features[25] * 0.227335f;
    z_63 += features[26] * -0.312895f;
    z_63 += features[27] * 0.226258f;
    z_63 += features[28] * 0.203140f;
    z_63 += features[29] * -0.133563f;
    z_63 += features[31] * -0.219842f;
    z_63 += features[33] * 0.112230f;
    z_63 += features[36] * -0.198557f;
    z_63 += features[39] * -0.232096f;
    z_63 += features[41] * -0.123837f;
    z_63 += features[44] * 0.116809f;
    z_63 += features[47] * 0.226677f;
    z_63 += features[48] * -0.185496f;
    z_63 += features[49] * -0.269248f;
    z_63 += features[50] * 0.211402f;
    z_63 += features[53] * -0.158800f;
    z_63 += features[55] * 0.260678f;
    z_63 += features[56] * 0.121436f;
    z_63 += features[58] * 0.289179f;
    z_63 += features[61] * 0.234154f;
    float out_63 = 0.230402f * z_63;
    {
        float arg_x = 0.151680f * z_63 + -0.047543f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.162870f * z_63 + 0.060854f) + 1e-6f;
        out_63 += 0.023677f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.132789f * z_63 + -0.015736f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.112909f * z_63 + 0.026371f) + 1e-6f;
        out_63 += 0.018420f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.192091f * z_63 + -0.031504f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.179402f * z_63 + 0.059701f) + 1e-6f;
        out_63 += 0.038312f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.080826f * z_63 + -0.089659f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.102546f * z_63 + 0.088041f) + 1e-6f;
        out_63 += -0.011898f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[63] = out_63;

    // Node 64
    float z_64 = 0.0f;
    z_64 += features[0] * 0.165774f;
    z_64 += features[2] * -0.135797f;
    z_64 += features[4] * 0.186174f;
    z_64 += features[5] * -0.137335f;
    z_64 += features[7] * 0.135153f;
    z_64 += features[8] * 0.132155f;
    z_64 += features[10] * 0.222852f;
    z_64 += features[13] * -0.140592f;
    z_64 += features[14] * -0.169300f;
    z_64 += features[18] * 0.226151f;
    z_64 += features[19] * -0.233412f;
    z_64 += features[20] * -0.141773f;
    z_64 += features[23] * -0.176103f;
    z_64 += features[24] * -0.147791f;
    z_64 += features[26] * -0.273183f;
    z_64 += features[27] * 0.233164f;
    z_64 += features[29] * -0.152299f;
    z_64 += features[30] * 0.179122f;
    z_64 += features[31] * -0.172970f;
    z_64 += features[32] * -0.163577f;
    z_64 += features[34] * -0.135472f;
    z_64 += features[36] * -0.178341f;
    z_64 += features[37] * -0.115322f;
    z_64 += features[38] * 0.124420f;
    z_64 += features[39] * -0.192393f;
    z_64 += features[40] * -0.239134f;
    z_64 += features[44] * -0.122282f;
    z_64 += features[45] * 0.185336f;
    z_64 += features[47] * 0.151249f;
    z_64 += features[48] * -0.194866f;
    z_64 += features[49] * -0.147596f;
    z_64 += features[50] * 0.204142f;
    z_64 += features[51] * -0.169453f;
    z_64 += features[54] * 0.276982f;
    z_64 += features[56] * 0.125425f;
    z_64 += features[62] * 0.140795f;
    z_64 += features[63] * 0.188049f;
    float out_64 = 0.225483f * z_64;
    {
        float arg_x = -0.143810f * z_64 + -0.039954f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.097990f * z_64 + 0.054121f) + 1e-6f;
        out_64 += 0.004174f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.176972f * z_64 + -0.027588f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.104202f * z_64 + 0.053345f) + 1e-6f;
        out_64 += -0.012821f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.110089f * z_64 + 0.036511f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.158710f * z_64 + -0.032588f) + 1e-6f;
        out_64 += 0.076981f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.081980f * z_64 + 0.021633f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.102909f * z_64 + -0.024065f) + 1e-6f;
        out_64 += 0.060512f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[64] = out_64;

    // Node 65
    float z_65 = 0.0f;
    z_65 += features[0] * -0.115075f;
    z_65 += features[1] * -0.214382f;
    z_65 += features[3] * 0.128337f;
    z_65 += features[4] * -0.159656f;
    z_65 += features[6] * 0.147046f;
    z_65 += features[7] * 0.168915f;
    z_65 += features[12] * 0.169632f;
    z_65 += features[14] * -0.228997f;
    z_65 += features[15] * -0.134316f;
    z_65 += features[16] * 0.138438f;
    z_65 += features[17] * -0.273707f;
    z_65 += features[19] * -0.110576f;
    z_65 += features[20] * -0.183615f;
    z_65 += features[21] * 0.153922f;
    z_65 += features[22] * -0.110687f;
    z_65 += features[23] * -0.161947f;
    z_65 += features[26] * -0.187446f;
    z_65 += features[27] * 0.155950f;
    z_65 += features[29] * 0.159561f;
    z_65 += features[30] * -0.139546f;
    z_65 += features[32] * 0.179303f;
    z_65 += features[38] * 0.268511f;
    z_65 += features[39] * -0.150027f;
    z_65 += features[40] * -0.237022f;
    z_65 += features[42] * -0.150345f;
    z_65 += features[44] * 0.179874f;
    z_65 += features[45] * -0.285545f;
    z_65 += features[46] * 0.111961f;
    z_65 += features[48] * 0.216269f;
    z_65 += features[50] * -0.141153f;
    z_65 += features[51] * -0.110742f;
    z_65 += features[60] * 0.141748f;
    z_65 += features[61] * 0.223529f;
    z_65 += features[62] * 0.171497f;
    z_65 += features[63] * -0.123754f;
    float out_65 = 0.251342f * z_65;
    {
        float arg_x = 0.143707f * z_65 + -0.025795f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.133374f * z_65 + 0.039187f) + 1e-6f;
        out_65 += 0.033974f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.114729f * z_65 + -0.051763f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.121664f * z_65 + 0.059357f) + 1e-6f;
        out_65 += 0.018019f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.128198f * z_65 + -0.056096f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.099536f * z_65 + 0.069383f) + 1e-6f;
        out_65 += 0.015246f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.097181f * z_65 + -0.084929f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.116121f * z_65 + 0.089640f) + 1e-6f;
        out_65 += 0.010599f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[65] = out_65;

    // Node 66
    float z_66 = 0.0f;
    z_66 += features[4] * -0.132246f;
    z_66 += features[6] * -0.137630f;
    z_66 += features[8] * -0.130931f;
    z_66 += features[10] * 0.186814f;
    z_66 += features[12] * -0.142402f;
    z_66 += features[14] * 0.129622f;
    z_66 += features[17] * -0.240040f;
    z_66 += features[19] * 0.226748f;
    z_66 += features[21] * -0.181520f;
    z_66 += features[26] * 0.109969f;
    z_66 += features[27] * 0.169522f;
    z_66 += features[29] * 0.145832f;
    z_66 += features[30] * -0.141297f;
    z_66 += features[31] * 0.202425f;
    z_66 += features[32] * 0.227717f;
    z_66 += features[35] * 0.127068f;
    z_66 += features[37] * 0.203198f;
    z_66 += features[40] * 0.163664f;
    z_66 += features[42] * -0.238436f;
    z_66 += features[44] * 0.162951f;
    z_66 += features[45] * -0.256234f;
    z_66 += features[46] * -0.254827f;
    z_66 += features[47] * -0.197375f;
    z_66 += features[53] * -0.181474f;
    z_66 += features[54] * -0.123664f;
    z_66 += features[55] * 0.153344f;
    z_66 += features[57] * -0.118436f;
    z_66 += features[61] * -0.195784f;
    z_66 += features[63] * 0.210086f;
    float out_66 = 0.199499f * z_66;
    {
        float arg_x = -0.046313f * z_66 + -0.038119f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.064245f * z_66 + 0.038127f) + 1e-6f;
        out_66 += -0.048495f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.083455f * z_66 + 0.007923f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.067144f * z_66 + 0.005197f) + 1e-6f;
        out_66 += -0.065011f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.077573f * z_66 + 0.057347f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.102886f * z_66 + -0.055249f) + 1e-6f;
        out_66 += -0.089074f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.080738f * z_66 + 0.008470f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.075994f * z_66 + 0.002235f) + 1e-6f;
        out_66 += -0.066676f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[66] = out_66;

    // Node 67
    float z_67 = 0.0f;
    z_67 += features[1] * -0.227536f;
    z_67 += features[7] * 0.278833f;
    z_67 += features[11] * -0.210194f;
    z_67 += features[12] * -0.169059f;
    z_67 += features[15] * -0.131573f;
    z_67 += features[23] * -0.148411f;
    z_67 += features[24] * -0.224405f;
    z_67 += features[26] * -0.244043f;
    z_67 += features[27] * 0.262974f;
    z_67 += features[30] * 0.193339f;
    z_67 += features[31] * 0.145764f;
    z_67 += features[32] * -0.241259f;
    z_67 += features[34] * 0.266048f;
    z_67 += features[35] * 0.142765f;
    z_67 += features[36] * -0.138656f;
    z_67 += features[38] * 0.298397f;
    z_67 += features[39] * 0.179901f;
    z_67 += features[42] * -0.209815f;
    z_67 += features[43] * -0.148761f;
    z_67 += features[44] * 0.212541f;
    z_67 += features[46] * 0.149761f;
    z_67 += features[47] * -0.111942f;
    z_67 += features[49] * -0.260626f;
    z_67 += features[50] * -0.126959f;
    z_67 += features[51] * -0.131804f;
    z_67 += features[52] * 0.134644f;
    z_67 += features[56] * 0.157802f;
    z_67 += features[57] * 0.230169f;
    z_67 += features[58] * 0.245800f;
    z_67 += features[59] * -0.160102f;
    z_67 += features[60] * 0.136734f;
    z_67 += features[61] * -0.196422f;
    z_67 += features[62] * 0.117083f;
    z_67 += features[63] * 0.211277f;
    float out_67 = 0.235579f * z_67;
    {
        float arg_x = -0.184892f * z_67 + 0.010022f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.154245f * z_67 + 0.017249f) + 1e-6f;
        out_67 += -0.055110f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.207072f * z_67 + 0.015529f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.149339f * z_67 + 0.021591f) + 1e-6f;
        out_67 += -0.047964f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.124517f * z_67 + -0.044409f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.102662f * z_67 + 0.045101f) + 1e-6f;
        out_67 += -0.016049f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.203646f * z_67 + 0.019145f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.153510f * z_67 + 0.017722f) + 1e-6f;
        out_67 += -0.055600f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[67] = out_67;

    // Node 68
    float z_68 = 0.0f;
    z_68 += features[1] * -0.167325f;
    z_68 += features[3] * 0.116708f;
    z_68 += features[5] * 0.124460f;
    z_68 += features[6] * 0.155432f;
    z_68 += features[7] * 0.216124f;
    z_68 += features[10] * 0.168973f;
    z_68 += features[13] * 0.137027f;
    z_68 += features[17] * -0.213486f;
    z_68 += features[20] * -0.151467f;
    z_68 += features[23] * -0.279240f;
    z_68 += features[24] * -0.122570f;
    z_68 += features[27] * 0.120539f;
    z_68 += features[32] * -0.151851f;
    z_68 += features[33] * 0.146240f;
    z_68 += features[36] * -0.219370f;
    z_68 += features[38] * 0.197542f;
    z_68 += features[41] * -0.159325f;
    z_68 += features[42] * -0.203336f;
    z_68 += features[44] * 0.158222f;
    z_68 += features[49] * -0.167911f;
    z_68 += features[50] * -0.141183f;
    z_68 += features[51] * -0.176981f;
    z_68 += features[52] * 0.136648f;
    z_68 += features[54] * -0.218721f;
    z_68 += features[55] * -0.133958f;
    z_68 += features[56] * 0.178185f;
    z_68 += features[57] * 0.151239f;
    z_68 += features[61] * -0.228139f;
    z_68 += features[62] * -0.208258f;
    z_68 += features[63] * 0.146894f;
    float out_68 = 0.203218f * z_68;
    {
        float arg_x = 0.079166f * z_68 + -0.091040f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.084294f * z_68 + 0.091035f) + 1e-6f;
        out_68 += -0.031809f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.098490f * z_68 + -0.102036f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.080980f * z_68 + 0.103121f) + 1e-6f;
        out_68 += -0.023310f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.058524f * z_68 + -0.086290f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.084795f * z_68 + 0.084379f) + 1e-6f;
        out_68 += -0.035216f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.096715f * z_68 + 0.084215f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.116232f * z_68 + -0.088069f) + 1e-6f;
        out_68 += -0.102039f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[68] = out_68;

    // Node 69
    float z_69 = 0.0f;
    z_69 += features[4] * 0.130416f;
    z_69 += features[5] * 0.177740f;
    z_69 += features[8] * 0.223523f;
    z_69 += features[9] * -0.198861f;
    z_69 += features[14] * -0.262666f;
    z_69 += features[16] * -0.185670f;
    z_69 += features[17] * 0.141287f;
    z_69 += features[18] * 0.113652f;
    z_69 += features[20] * 0.133031f;
    z_69 += features[21] * 0.318354f;
    z_69 += features[22] * 0.191509f;
    z_69 += features[23] * -0.141054f;
    z_69 += features[28] * -0.175588f;
    z_69 += features[30] * 0.228785f;
    z_69 += features[31] * -0.137703f;
    z_69 += features[32] * -0.232888f;
    z_69 += features[35] * -0.184071f;
    z_69 += features[37] * -0.128449f;
    z_69 += features[38] * -0.156167f;
    z_69 += features[41] * 0.220516f;
    z_69 += features[44] * -0.272751f;
    z_69 += features[46] * 0.262597f;
    z_69 += features[47] * 0.182092f;
    z_69 += features[48] * -0.247842f;
    z_69 += features[52] * -0.114473f;
    z_69 += features[53] * 0.115487f;
    z_69 += features[56] * 0.197350f;
    z_69 += features[57] * 0.232733f;
    z_69 += features[62] * 0.126343f;
    float out_69 = 0.260745f * z_69;
    {
        float arg_x = -0.167395f * z_69 + -0.042228f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.136664f * z_69 + 0.056284f) + 1e-6f;
        out_69 += 0.002441f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.242385f * z_69 + -0.016331f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.170973f * z_69 + 0.050761f) + 1e-6f;
        out_69 += -0.036568f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.183159f * z_69 + -0.036885f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.139066f * z_69 + 0.054809f) + 1e-6f;
        out_69 += -0.005471f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.162528f * z_69 + -0.044891f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.154819f * z_69 + 0.057195f) + 1e-6f;
        out_69 += 0.003198f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[69] = out_69;

    // Node 70
    float z_70 = 0.0f;
    z_70 += features[0] * 0.231221f;
    z_70 += features[2] * 0.135492f;
    z_70 += features[6] * 0.155938f;
    z_70 += features[8] * 0.176009f;
    z_70 += features[9] * -0.196764f;
    z_70 += features[12] * -0.240509f;
    z_70 += features[13] * -0.160039f;
    z_70 += features[17] * 0.186542f;
    z_70 += features[18] * 0.139467f;
    z_70 += features[20] * -0.190573f;
    z_70 += features[22] * 0.191240f;
    z_70 += features[23] * -0.164963f;
    z_70 += features[25] * -0.247345f;
    z_70 += features[30] * 0.130552f;
    z_70 += features[32] * -0.142708f;
    z_70 += features[33] * -0.211648f;
    z_70 += features[37] * -0.156995f;
    z_70 += features[41] * 0.165006f;
    z_70 += features[43] * -0.127483f;
    z_70 += features[47] * 0.200601f;
    z_70 += features[49] * -0.257480f;
    z_70 += features[53] * 0.205321f;
    z_70 += features[55] * -0.186044f;
    z_70 += features[56] * -0.152119f;
    z_70 += features[58] * -0.204242f;
    z_70 += features[59] * -0.180483f;
    z_70 += features[60] * 0.251883f;
    z_70 += features[62] * 0.124071f;
    float out_70 = 0.201332f * z_70;
    {
        float arg_x = -0.054400f * z_70 + -0.024760f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.112504f * z_70 + 0.016948f) + 1e-6f;
        out_70 += -0.022066f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.112339f * z_70 + 0.012528f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.073778f * z_70 + 0.011830f) + 1e-6f;
        out_70 += -0.041243f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.117790f * z_70 + 0.012988f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.070379f * z_70 + 0.017009f) + 1e-6f;
        out_70 += -0.033979f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.073753f * z_70 + -0.048048f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.077549f * z_70 + 0.052333f) + 1e-6f;
        out_70 += -0.008755f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[70] = out_70;

    // Node 71
    float z_71 = 0.0f;
    z_71 += features[5] * 0.129325f;
    z_71 += features[7] * 0.213342f;
    z_71 += features[8] * -0.158808f;
    z_71 += features[9] * 0.123846f;
    z_71 += features[10] * 0.169581f;
    z_71 += features[11] * 0.276563f;
    z_71 += features[14] * 0.233928f;
    z_71 += features[16] * -0.157376f;
    z_71 += features[18] * -0.159206f;
    z_71 += features[19] * 0.134484f;
    z_71 += features[20] * -0.213825f;
    z_71 += features[21] * -0.222748f;
    z_71 += features[24] * 0.124479f;
    z_71 += features[27] * 0.200679f;
    z_71 += features[28] * 0.193558f;
    z_71 += features[29] * -0.197989f;
    z_71 += features[30] * -0.127732f;
    z_71 += features[31] * 0.116835f;
    z_71 += features[32] * 0.169467f;
    z_71 += features[38] * 0.156509f;
    z_71 += features[41] * -0.255643f;
    z_71 += features[42] * -0.125262f;
    z_71 += features[43] * 0.159221f;
    z_71 += features[44] * 0.201301f;
    z_71 += features[45] * -0.270000f;
    z_71 += features[47] * -0.136747f;
    z_71 += features[49] * 0.168572f;
    z_71 += features[50] * -0.204206f;
    z_71 += features[51] * 0.121314f;
    z_71 += features[52] * 0.189148f;
    z_71 += features[57] * -0.207293f;
    z_71 += features[58] * 0.167305f;
    z_71 += features[60] * -0.158515f;
    z_71 += features[61] * -0.156473f;
    z_71 += features[63] * 0.174328f;
    float out_71 = 0.225966f * z_71;
    {
        float arg_x = -0.150210f * z_71 + 0.003564f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.102007f * z_71 + 0.013732f) + 1e-6f;
        out_71 += -0.029808f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.151525f * z_71 + -0.000856f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.161537f * z_71 + 0.012219f) + 1e-6f;
        out_71 += -0.033705f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.105224f * z_71 + -0.025265f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.141599f * z_71 + 0.030335f) + 1e-6f;
        out_71 += 0.043960f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.107758f * z_71 + -0.010824f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.113390f * z_71 + 0.011563f) + 1e-6f;
        out_71 += -0.025434f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[71] = out_71;

    // Node 72
    float z_72 = 0.0f;
    z_72 += features[0] * -0.187703f;
    z_72 += features[1] * -0.309284f;
    z_72 += features[3] * -0.230436f;
    z_72 += features[5] * -0.240130f;
    z_72 += features[7] * 0.269550f;
    z_72 += features[8] * -0.175463f;
    z_72 += features[9] * -0.142381f;
    z_72 += features[10] * 0.111121f;
    z_72 += features[11] * -0.168938f;
    z_72 += features[12] * 0.188037f;
    z_72 += features[13] * 0.198961f;
    z_72 += features[14] * -0.184560f;
    z_72 += features[16] * 0.272080f;
    z_72 += features[19] * -0.116750f;
    z_72 += features[20] * -0.263290f;
    z_72 += features[21] * 0.110573f;
    z_72 += features[23] * 0.117067f;
    z_72 += features[25] * -0.163813f;
    z_72 += features[27] * -0.162916f;
    z_72 += features[29] * 0.233867f;
    z_72 += features[32] * 0.153006f;
    z_72 += features[34] * -0.130868f;
    z_72 += features[38] * 0.212488f;
    z_72 += features[39] * 0.134305f;
    z_72 += features[40] * -0.171126f;
    z_72 += features[41] * 0.228412f;
    z_72 += features[43] * 0.142231f;
    z_72 += features[45] * -0.169553f;
    z_72 += features[48] * 0.250800f;
    z_72 += features[49] * -0.245698f;
    z_72 += features[52] * -0.187427f;
    z_72 += features[53] * -0.301218f;
    z_72 += features[55] * 0.261802f;
    z_72 += features[57] * 0.147640f;
    z_72 += features[58] * -0.177053f;
    z_72 += features[60] * 0.115629f;
    z_72 += features[61] * 0.206356f;
    z_72 += features[62] * 0.240146f;
    float out_72 = 0.239756f * z_72;
    {
        float arg_x = -0.110936f * z_72 + -0.014453f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.178770f * z_72 + 0.016732f) + 1e-6f;
        out_72 += -0.043515f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.107209f * z_72 + -0.048441f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.093837f * z_72 + 0.067094f) + 1e-6f;
        out_72 += -0.019404f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.147054f * z_72 + -0.008289f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.169789f * z_72 + 0.017490f) + 1e-6f;
        out_72 += 0.037434f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.184901f * z_72 + 0.002523f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.173095f * z_72 + 0.020657f) + 1e-6f;
        out_72 += 0.038133f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[72] = out_72;

    // Node 73
    float z_73 = 0.0f;
    z_73 += features[3] * 0.291553f;
    z_73 += features[4] * 0.134003f;
    z_73 += features[5] * 0.189192f;
    z_73 += features[6] * -0.126445f;
    z_73 += features[7] * -0.144514f;
    z_73 += features[8] * 0.166411f;
    z_73 += features[10] * 0.204768f;
    z_73 += features[11] * 0.213088f;
    z_73 += features[14] * -0.173112f;
    z_73 += features[16] * -0.121972f;
    z_73 += features[17] * 0.160493f;
    z_73 += features[18] * 0.126064f;
    z_73 += features[19] * 0.152942f;
    z_73 += features[20] * 0.171837f;
    z_73 += features[22] * -0.193907f;
    z_73 += features[26] * 0.265695f;
    z_73 += features[28] * 0.185467f;
    z_73 += features[29] * -0.112626f;
    z_73 += features[33] * 0.192885f;
    z_73 += features[35] * -0.230137f;
    z_73 += features[38] * -0.261225f;
    z_73 += features[39] * -0.138389f;
    z_73 += features[41] * -0.325790f;
    z_73 += features[42] * 0.145798f;
    z_73 += features[43] * -0.180606f;
    z_73 += features[53] * 0.285197f;
    z_73 += features[55] * -0.191583f;
    z_73 += features[56] * 0.221196f;
    z_73 += features[57] * -0.117559f;
    z_73 += features[60] * -0.194937f;
    z_73 += features[62] * -0.304188f;
    float out_73 = 0.225766f * z_73;
    {
        float arg_x = -0.023691f * z_73 + -0.137043f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.020605f * z_73 + 0.130726f) + 1e-6f;
        out_73 += 0.017637f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.053782f * z_73 + -0.075768f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.047416f * z_73 + 0.072863f) + 1e-6f;
        out_73 += 0.027284f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.023751f * z_73 + -0.179512f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.000813f * z_73 + 0.173806f) + 1e-6f;
        out_73 += 0.017373f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.180648f * z_73 + -0.061633f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.179763f * z_73 + 0.072198f) + 1e-6f;
        out_73 += -0.010641f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[73] = out_73;

    // Node 74
    float z_74 = 0.0f;
    z_74 += features[1] * 0.170239f;
    z_74 += features[3] * 0.179710f;
    z_74 += features[6] * -0.155063f;
    z_74 += features[8] * -0.206383f;
    z_74 += features[9] * 0.201427f;
    z_74 += features[10] * 0.212801f;
    z_74 += features[12] * 0.132534f;
    z_74 += features[13] * 0.116492f;
    z_74 += features[15] * -0.246448f;
    z_74 += features[17] * -0.139416f;
    z_74 += features[18] * 0.125837f;
    z_74 += features[19] * -0.148205f;
    z_74 += features[20] * 0.220579f;
    z_74 += features[23] * 0.204989f;
    z_74 += features[29] * -0.174081f;
    z_74 += features[33] * 0.169846f;
    z_74 += features[36] * -0.199792f;
    z_74 += features[38] * 0.139909f;
    z_74 += features[40] * 0.156223f;
    z_74 += features[42] * -0.141818f;
    z_74 += features[45] * 0.121324f;
    z_74 += features[46] * 0.185675f;
    z_74 += features[48] * -0.204211f;
    z_74 += features[49] * 0.157338f;
    z_74 += features[53] * -0.158804f;
    z_74 += features[54] * -0.193885f;
    z_74 += features[56] * 0.127363f;
    z_74 += features[58] * 0.226467f;
    z_74 += features[59] * 0.206229f;
    z_74 += features[60] * -0.218434f;
    z_74 += features[63] * 0.171800f;
    float out_74 = 0.208326f * z_74;
    {
        float arg_x = -0.144621f * z_74 + -0.024862f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.108055f * z_74 + 0.044331f) + 1e-6f;
        out_74 += -0.031687f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.109569f * z_74 + 0.026748f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.093114f * z_74 + -0.006895f) + 1e-6f;
        out_74 += 0.047025f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.134661f * z_74 + -0.131852f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.190255f * z_74 + 0.127910f) + 1e-6f;
        out_74 += -0.000811f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.140721f * z_74 + -0.080640f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.157113f * z_74 + 0.094281f) + 1e-6f;
        out_74 += -0.016890f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[74] = out_74;

    // Node 75
    float z_75 = 0.0f;
    z_75 += features[2] * 0.141624f;
    z_75 += features[3] * -0.122801f;
    z_75 += features[4] * -0.147709f;
    z_75 += features[8] * -0.168323f;
    z_75 += features[10] * 0.133956f;
    z_75 += features[12] * 0.266112f;
    z_75 += features[13] * 0.222483f;
    z_75 += features[14] * 0.175626f;
    z_75 += features[16] * 0.191294f;
    z_75 += features[18] * -0.179459f;
    z_75 += features[21] * -0.144562f;
    z_75 += features[26] * -0.138180f;
    z_75 += features[29] * 0.286641f;
    z_75 += features[32] * 0.138805f;
    z_75 += features[33] * 0.126275f;
    z_75 += features[35] * 0.126264f;
    z_75 += features[38] * 0.246991f;
    z_75 += features[42] * -0.162457f;
    z_75 += features[43] * 0.194320f;
    z_75 += features[44] * 0.210878f;
    z_75 += features[46] * -0.118261f;
    z_75 += features[48] * 0.181834f;
    z_75 += features[52] * -0.145557f;
    z_75 += features[53] * -0.256194f;
    z_75 += features[54] * -0.109975f;
    z_75 += features[56] * 0.109812f;
    z_75 += features[59] * 0.116047f;
    z_75 += features[61] * 0.172487f;
    float out_75 = 0.187900f * z_75;
    {
        float arg_x = 0.138612f * z_75 + -0.089493f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.121598f * z_75 + 0.095287f) + 1e-6f;
        out_75 += 0.004208f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.165398f * z_75 + -0.009191f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.100714f * z_75 + 0.025367f) + 1e-6f;
        out_75 += 0.024522f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.119976f * z_75 + -0.030808f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.069462f * z_75 + 0.035532f) + 1e-6f;
        out_75 += 0.011651f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.102213f * z_75 + -0.008256f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.130606f * z_75 + 0.006655f) + 1e-6f;
        out_75 += 0.055252f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[75] = out_75;

    // Node 76
    float z_76 = 0.0f;
    z_76 += features[1] * 0.117546f;
    z_76 += features[3] * 0.253674f;
    z_76 += features[4] * -0.334407f;
    z_76 += features[5] * 0.179721f;
    z_76 += features[6] * -0.144855f;
    z_76 += features[9] * 0.183445f;
    z_76 += features[10] * 0.128171f;
    z_76 += features[12] * 0.135145f;
    z_76 += features[14] * 0.148483f;
    z_76 += features[16] * 0.279549f;
    z_76 += features[17] * -0.160946f;
    z_76 += features[18] * -0.241679f;
    z_76 += features[19] * 0.256239f;
    z_76 += features[20] * -0.146719f;
    z_76 += features[21] * -0.269408f;
    z_76 += features[22] * -0.293095f;
    z_76 += features[23] * -0.118339f;
    z_76 += features[26] * 0.150598f;
    z_76 += features[29] * 0.193710f;
    z_76 += features[30] * -0.246811f;
    z_76 += features[31] * 0.239395f;
    z_76 += features[32] * 0.192683f;
    z_76 += features[34] * 0.246985f;
    z_76 += features[35] * 0.203073f;
    z_76 += features[41] * -0.148387f;
    z_76 += features[42] * -0.176269f;
    z_76 += features[43] * 0.123778f;
    z_76 += features[44] * 0.221326f;
    z_76 += features[45] * -0.308415f;
    z_76 += features[46] * -0.280858f;
    z_76 += features[48] * 0.217162f;
    z_76 += features[49] * 0.240832f;
    z_76 += features[53] * -0.243930f;
    z_76 += features[54] * -0.253350f;
    z_76 += features[58] * 0.159108f;
    z_76 += features[59] * 0.172994f;
    z_76 += features[61] * -0.190604f;
    z_76 += features[62] * -0.182109f;
    z_76 += features[63] * 0.137871f;
    float out_76 = 0.269935f * z_76;
    {
        float arg_x = 0.135349f * z_76 + -0.076616f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.097959f * z_76 + 0.094562f) + 1e-6f;
        out_76 += -0.005067f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.188679f * z_76 + -0.002089f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.202358f * z_76 + 0.030296f) + 1e-6f;
        out_76 += 0.039595f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.197534f * z_76 + 0.023438f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.165878f * z_76 + 0.019763f) + 1e-6f;
        out_76 += 0.039181f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.213454f * z_76 + 0.042741f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.194351f * z_76 + 0.006363f) + 1e-6f;
        out_76 += 0.067374f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[76] = out_76;

    // Node 77
    float z_77 = 0.0f;
    z_77 += features[1] * 0.165106f;
    z_77 += features[2] * 0.213083f;
    z_77 += features[3] * 0.157253f;
    z_77 += features[4] * -0.215215f;
    z_77 += features[11] * -0.238097f;
    z_77 += features[12] * -0.241779f;
    z_77 += features[13] * -0.208964f;
    z_77 += features[14] * 0.189554f;
    z_77 += features[15] * -0.125497f;
    z_77 += features[17] * 0.118388f;
    z_77 += features[18] * -0.214349f;
    z_77 += features[19] * 0.220170f;
    z_77 += features[20] * 0.199672f;
    z_77 += features[21] * -0.209616f;
    z_77 += features[22] * -0.154159f;
    z_77 += features[24] * -0.152769f;
    z_77 += features[25] * -0.151431f;
    z_77 += features[27] * 0.245134f;
    z_77 += features[28] * -0.241133f;
    z_77 += features[31] * 0.258408f;
    z_77 += features[32] * 0.158731f;
    z_77 += features[33] * 0.133066f;
    z_77 += features[34] * 0.238730f;
    z_77 += features[35] * 0.242040f;
    z_77 += features[36] * 0.223526f;
    z_77 += features[38] * 0.131678f;
    z_77 += features[43] * -0.142437f;
    z_77 += features[45] * -0.162352f;
    z_77 += features[46] * -0.312545f;
    z_77 += features[47] * -0.186506f;
    z_77 += features[53] * 0.235704f;
    z_77 += features[55] * -0.113621f;
    z_77 += features[58] * 0.198926f;
    z_77 += features[59] * 0.111105f;
    z_77 += features[61] * -0.254391f;
    z_77 += features[63] * 0.279116f;
    float out_77 = 0.250183f * z_77;
    {
        float arg_x = -0.214831f * z_77 + 0.170869f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.172589f * z_77 + -0.074863f) + 1e-6f;
        out_77 += -0.097698f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.024356f * z_77 + 0.027798f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.071125f * z_77 + -0.028307f) + 1e-6f;
        out_77 += -0.047614f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.185732f * z_77 + 0.009869f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.181116f * z_77 + 0.056020f) + 1e-6f;
        out_77 += 0.051805f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.166337f * z_77 + -0.017178f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.173155f * z_77 + 0.066572f) + 1e-6f;
        out_77 += 0.035666f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[77] = out_77;

    // Node 78
    float z_78 = 0.0f;
    z_78 += features[0] * 0.209431f;
    z_78 += features[1] * 0.127709f;
    z_78 += features[2] * 0.186544f;
    z_78 += features[3] * 0.124096f;
    z_78 += features[4] * -0.181635f;
    z_78 += features[10] * -0.212420f;
    z_78 += features[13] * -0.144656f;
    z_78 += features[15] * 0.116328f;
    z_78 += features[16] * -0.197621f;
    z_78 += features[17] * 0.184061f;
    z_78 += features[18] * -0.126971f;
    z_78 += features[19] * -0.163898f;
    z_78 += features[20] * 0.190080f;
    z_78 += features[23] * -0.164355f;
    z_78 += features[24] * -0.136932f;
    z_78 += features[26] * 0.109894f;
    z_78 += features[28] * -0.192980f;
    z_78 += features[29] * -0.238848f;
    z_78 += features[30] * 0.172631f;
    z_78 += features[31] * 0.128021f;
    z_78 += features[34] * 0.223547f;
    z_78 += features[35] * 0.119728f;
    z_78 += features[36] * 0.122224f;
    z_78 += features[37] * -0.238394f;
    z_78 += features[38] * -0.126796f;
    z_78 += features[42] * -0.143079f;
    z_78 += features[44] * -0.272112f;
    z_78 += features[46] * 0.152611f;
    z_78 += features[48] * -0.188858f;
    z_78 += features[50] * -0.208373f;
    z_78 += features[51] * -0.129951f;
    z_78 += features[53] * 0.285120f;
    z_78 += features[54] * -0.114616f;
    z_78 += features[56] * 0.200892f;
    z_78 += features[58] * -0.119336f;
    z_78 += features[60] * 0.123874f;
    z_78 += features[61] * -0.209937f;
    z_78 += features[62] * 0.134510f;
    float out_78 = 0.226116f * z_78;
    {
        float arg_x = -0.099744f * z_78 + -0.099890f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.061652f * z_78 + 0.098969f) + 1e-6f;
        out_78 += 0.033972f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.116661f * z_78 + -0.060682f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.116371f * z_78 + 0.063450f) + 1e-6f;
        out_78 += 0.013212f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.128622f * z_78 + -0.058885f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.115706f * z_78 + 0.062878f) + 1e-6f;
        out_78 += 0.009957f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.104534f * z_78 + -0.066311f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.086548f * z_78 + 0.067693f) + 1e-6f;
        out_78 += 0.025630f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[78] = out_78;

    // Node 79
    float z_79 = 0.0f;
    z_79 += features[0] * 0.199660f;
    z_79 += features[1] * 0.156216f;
    z_79 += features[2] * 0.118707f;
    z_79 += features[4] * 0.197994f;
    z_79 += features[5] * 0.234492f;
    z_79 += features[7] * -0.242703f;
    z_79 += features[11] * 0.184302f;
    z_79 += features[12] * -0.121279f;
    z_79 += features[14] * 0.115015f;
    z_79 += features[16] * -0.267915f;
    z_79 += features[17] * 0.197673f;
    z_79 += features[20] * -0.215640f;
    z_79 += features[23] * -0.148277f;
    z_79 += features[24] * 0.160967f;
    z_79 += features[25] * -0.175843f;
    z_79 += features[28] * -0.180018f;
    z_79 += features[29] * -0.313778f;
    z_79 += features[32] * -0.158982f;
    z_79 += features[34] * 0.179240f;
    z_79 += features[35] * -0.224303f;
    z_79 += features[36] * -0.126237f;
    z_79 += features[38] * -0.273910f;
    z_79 += features[46] * 0.127870f;
    z_79 += features[47] * 0.110243f;
    z_79 += features[48] * -0.199797f;
    z_79 += features[50] * 0.259169f;
    z_79 += features[51] * 0.144793f;
    z_79 += features[53] * 0.236597f;
    z_79 += features[54] * 0.193489f;
    z_79 += features[56] * -0.162935f;
    z_79 += features[59] * -0.223896f;
    z_79 += features[62] * -0.254303f;
    z_79 += features[63] * 0.213675f;
    float out_79 = 0.239986f * z_79;
    {
        float arg_x = 0.025551f * z_79 + -0.176638f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.025845f * z_79 + 0.175394f) + 1e-6f;
        out_79 += 0.011830f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.118248f * z_79 + 0.016565f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.165533f * z_79 + -0.015933f) + 1e-6f;
        out_79 += -0.102225f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.171398f * z_79 + 0.056323f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.142318f * z_79 + -0.006992f) + 1e-6f;
        out_79 += -0.082955f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.092975f * z_79 + -0.096335f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.068404f * z_79 + 0.106733f) + 1e-6f;
        out_79 += 0.023410f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[79] = out_79;

    // Node 80
    float z_80 = 0.0f;
    z_80 += features[0] * -0.215607f;
    z_80 += features[1] * -0.199859f;
    z_80 += features[3] * -0.202707f;
    z_80 += features[7] * 0.252557f;
    z_80 += features[9] * 0.180444f;
    z_80 += features[11] * 0.229883f;
    z_80 += features[15] * -0.186465f;
    z_80 += features[16] * 0.164206f;
    z_80 += features[19] * -0.134970f;
    z_80 += features[22] * -0.181321f;
    z_80 += features[25] * 0.136537f;
    z_80 += features[26] * -0.179544f;
    z_80 += features[28] * 0.132790f;
    z_80 += features[30] * -0.206574f;
    z_80 += features[32] * 0.179333f;
    z_80 += features[35] * -0.144920f;
    z_80 += features[36] * -0.167665f;
    z_80 += features[37] * 0.161211f;
    z_80 += features[38] * 0.205239f;
    z_80 += features[41] * -0.218159f;
    z_80 += features[42] * -0.146609f;
    z_80 += features[43] * 0.164974f;
    z_80 += features[44] * 0.165633f;
    z_80 += features[45] * -0.168168f;
    z_80 += features[53] * -0.190991f;
    z_80 += features[57] * -0.141921f;
    z_80 += features[58] * 0.166089f;
    z_80 += features[63] * 0.156623f;
    float out_80 = 0.194961f * z_80;
    {
        float arg_x = -0.039746f * z_80 + -0.004577f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.042330f * z_80 + -0.000567f) + 1e-6f;
        out_80 += 0.042596f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.181003f * z_80 + -0.089363f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.175422f * z_80 + 0.100193f) + 1e-6f;
        out_80 += -0.002371f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.167289f * z_80 + -0.085856f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.157111f * z_80 + 0.093666f) + 1e-6f;
        out_80 += 0.001995f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.078884f * z_80 + -0.027964f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.134179f * z_80 + 0.018988f) + 1e-6f;
        out_80 += 0.035304f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[80] = out_80;

    // Node 81
    float z_81 = 0.0f;
    z_81 += features[0] * -0.182013f;
    z_81 += features[4] * 0.132549f;
    z_81 += features[5] * 0.145552f;
    z_81 += features[10] * 0.125414f;
    z_81 += features[12] * 0.110560f;
    z_81 += features[13] * 0.152028f;
    z_81 += features[16] * -0.183723f;
    z_81 += features[19] * 0.162242f;
    z_81 += features[21] * -0.256837f;
    z_81 += features[23] * 0.161140f;
    z_81 += features[24] * 0.134165f;
    z_81 += features[29] * -0.124429f;
    z_81 += features[31] * -0.120344f;
    z_81 += features[41] * -0.142146f;
    z_81 += features[45] * 0.218145f;
    z_81 += features[50] * 0.144438f;
    z_81 += features[51] * 0.138895f;
    z_81 += features[56] * -0.174050f;
    z_81 += features[59] * 0.135355f;
    z_81 += features[62] * -0.166693f;
    float out_81 = 0.095908f * z_81;
    {
        float arg_x = 0.080547f * z_81 + -0.045593f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.075718f * z_81 + 0.045976f) + 1e-6f;
        out_81 += -0.031163f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.046780f * z_81 + -0.037380f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.033837f * z_81 + 0.035180f) + 1e-6f;
        out_81 += -0.032305f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.113215f * z_81 + -0.065505f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.106850f * z_81 + 0.069412f) + 1e-6f;
        out_81 += -0.018811f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.123280f * z_81 + -0.062242f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.143133f * z_81 + 0.066172f) + 1e-6f;
        out_81 += -0.022354f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[81] = out_81;

    // Node 82
    float z_82 = 0.0f;
    z_82 += features[1] * -0.246157f;
    z_82 += features[3] * -0.229142f;
    z_82 += features[5] * -0.110510f;
    z_82 += features[7] * 0.123900f;
    z_82 += features[12] * -0.174963f;
    z_82 += features[14] * -0.132355f;
    z_82 += features[16] * 0.169006f;
    z_82 += features[17] * -0.153069f;
    z_82 += features[18] * 0.112882f;
    z_82 += features[19] * -0.245066f;
    z_82 += features[20] * -0.142709f;
    z_82 += features[22] * 0.169072f;
    z_82 += features[23] * -0.159269f;
    z_82 += features[24] * -0.115954f;
    z_82 += features[26] * -0.115967f;
    z_82 += features[27] * 0.150274f;
    z_82 += features[31] * -0.119371f;
    z_82 += features[33] * -0.218866f;
    z_82 += features[35] * -0.138860f;
    z_82 += features[36] * -0.149116f;
    z_82 += features[42] * 0.149784f;
    z_82 += features[44] * -0.268256f;
    z_82 += features[45] * 0.132929f;
    z_82 += features[46] * 0.169539f;
    z_82 += features[47] * 0.170626f;
    z_82 += features[48] * -0.218798f;
    z_82 += features[49] * -0.134409f;
    z_82 += features[50] * 0.242794f;
    z_82 += features[51] * -0.135984f;
    z_82 += features[54] * 0.168513f;
    z_82 += features[55] * 0.228429f;
    z_82 += features[60] * 0.263736f;
    z_82 += features[62] * 0.191482f;
    float out_82 = 0.210818f * z_82;
    {
        float arg_x = -0.105432f * z_82 + 0.009780f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.127407f * z_82 + -0.009013f) + 1e-6f;
        out_82 += -0.047080f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.129749f * z_82 + -0.059202f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.102181f * z_82 + 0.065881f) + 1e-6f;
        out_82 += 0.013531f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.089313f * z_82 + -0.062509f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.112996f * z_82 + 0.063875f) + 1e-6f;
        out_82 += 0.011901f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.041807f * z_82 + -0.071312f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.051147f * z_82 + 0.070685f) + 1e-6f;
        out_82 += 0.000631f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[82] = out_82;

    // Node 83
    float z_83 = 0.0f;
    z_83 += features[1] * 0.189469f;
    z_83 += features[2] * 0.164384f;
    z_83 += features[6] * -0.198748f;
    z_83 += features[8] * -0.203706f;
    z_83 += features[9] * -0.115872f;
    z_83 += features[10] * -0.214427f;
    z_83 += features[13] * -0.132167f;
    z_83 += features[14] * 0.214110f;
    z_83 += features[20] * 0.110585f;
    z_83 += features[23] * -0.176630f;
    z_83 += features[26] * -0.187947f;
    z_83 += features[29] * 0.262336f;
    z_83 += features[31] * 0.207843f;
    z_83 += features[33] * -0.170481f;
    z_83 += features[34] * 0.240688f;
    z_83 += features[37] * 0.193307f;
    z_83 += features[41] * -0.164467f;
    z_83 += features[42] * -0.162898f;
    z_83 += features[43] * 0.151290f;
    z_83 += features[44] * -0.118998f;
    z_83 += features[46] * -0.255528f;
    z_83 += features[48] * 0.247314f;
    z_83 += features[56] * -0.215983f;
    z_83 += features[60] * 0.251995f;
    float out_83 = 0.181102f * z_83;
    {
        float arg_x = 0.124348f * z_83 + -0.026602f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.085090f * z_83 + 0.015443f) + 1e-6f;
        out_83 += 0.018142f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.027711f * z_83 + -0.117201f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.005667f * z_83 + 0.111462f) + 1e-6f;
        out_83 += -0.017356f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.036915f * z_83 + -0.100284f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.001510f * z_83 + 0.106135f) + 1e-6f;
        out_83 += 0.017406f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.122626f * z_83 + -0.032613f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.070613f * z_83 + 0.022148f) + 1e-6f;
        out_83 += 0.015012f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[83] = out_83;

    // Node 84
    float z_84 = 0.0f;
    z_84 += features[1] * 0.131553f;
    z_84 += features[3] * -0.169396f;
    z_84 += features[5] * -0.168224f;
    z_84 += features[10] * 0.216528f;
    z_84 += features[11] * -0.111586f;
    z_84 += features[17] * -0.220608f;
    z_84 += features[18] * -0.129657f;
    z_84 += features[23] * 0.142284f;
    z_84 += features[30] * -0.132501f;
    z_84 += features[31] * -0.183530f;
    z_84 += features[32] * 0.221992f;
    z_84 += features[33] * 0.204588f;
    z_84 += features[34] * -0.256524f;
    z_84 += features[37] * 0.215031f;
    z_84 += features[38] * 0.211259f;
    z_84 += features[39] * 0.147582f;
    z_84 += features[43] * 0.135239f;
    z_84 += features[44] * 0.193029f;
    z_84 += features[46] * -0.111078f;
    z_84 += features[47] * -0.215809f;
    z_84 += features[51] * 0.190744f;
    z_84 += features[52] * -0.165204f;
    z_84 += features[53] * -0.157069f;
    z_84 += features[56] * -0.165381f;
    z_84 += features[59] * 0.185752f;
    z_84 += features[60] * -0.110530f;
    float out_84 = 0.183056f * z_84;
    {
        float arg_x = -0.026174f * z_84 + -0.103283f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.023740f * z_84 + 0.103010f) + 1e-6f;
        out_84 += -0.027595f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.041606f * z_84 + -0.099964f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.030854f * z_84 + 0.104812f) + 1e-6f;
        out_84 += -0.031649f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.092511f * z_84 + -0.037559f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.088413f * z_84 + 0.037896f) + 1e-6f;
        out_84 += 0.014282f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.120585f * z_84 + -0.028044f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.087481f * z_84 + 0.034418f) + 1e-6f;
        out_84 += 0.021042f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[84] = out_84;

    // Node 85
    float z_85 = 0.0f;
    z_85 += features[7] * 0.219521f;
    z_85 += features[8] * -0.226903f;
    z_85 += features[11] * 0.282181f;
    z_85 += features[13] * 0.163179f;
    z_85 += features[15] * -0.169995f;
    z_85 += features[17] * -0.243025f;
    z_85 += features[20] * 0.123873f;
    z_85 += features[24] * -0.170450f;
    z_85 += features[27] * 0.166473f;
    z_85 += features[29] * -0.215630f;
    z_85 += features[34] * 0.187010f;
    z_85 += features[36] * -0.134676f;
    z_85 += features[37] * -0.177053f;
    z_85 += features[38] * 0.117415f;
    z_85 += features[41] * -0.284162f;
    z_85 += features[42] * -0.198611f;
    z_85 += features[44] * 0.139241f;
    z_85 += features[45] * -0.131791f;
    z_85 += features[48] * -0.194786f;
    z_85 += features[50] * -0.125269f;
    z_85 += features[53] * 0.185304f;
    z_85 += features[54] * -0.253412f;
    z_85 += features[56] * 0.143489f;
    z_85 += features[61] * -0.131108f;
    z_85 += features[62] * -0.174308f;
    float out_85 = 0.199772f * z_85;
    {
        float arg_x = -0.129509f * z_85 + -0.028219f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.139404f * z_85 + 0.034558f) + 1e-6f;
        out_85 += -0.033700f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.118915f * z_85 + 0.018820f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.093743f * z_85 + -0.011358f) + 1e-6f;
        out_85 += -0.056945f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.121852f * z_85 + 0.022199f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.138782f * z_85 + -0.016761f) + 1e-6f;
        out_85 += -0.064865f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.132752f * z_85 + -0.022798f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.153327f * z_85 + 0.029820f) + 1e-6f;
        out_85 += -0.037649f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[85] = out_85;

    // Node 86
    float z_86 = 0.0f;
    z_86 += features[0] * -0.113833f;
    z_86 += features[1] * -0.239835f;
    z_86 += features[2] * -0.140814f;
    z_86 += features[3] * 0.144934f;
    z_86 += features[5] * -0.122660f;
    z_86 += features[8] * 0.135356f;
    z_86 += features[10] * 0.260965f;
    z_86 += features[11] * -0.213112f;
    z_86 += features[12] * 0.117847f;
    z_86 += features[13] * 0.115931f;
    z_86 += features[15] * -0.217306f;
    z_86 += features[16] * 0.178455f;
    z_86 += features[20] * -0.167197f;
    z_86 += features[21] * -0.119244f;
    z_86 += features[22] * -0.210158f;
    z_86 += features[23] * 0.194774f;
    z_86 += features[24] * 0.145282f;
    z_86 += features[26] * -0.124623f;
    z_86 += features[31] * -0.213153f;
    z_86 += features[32] * 0.130681f;
    z_86 += features[38] * 0.286327f;
    z_86 += features[39] * -0.184208f;
    z_86 += features[47] * -0.167824f;
    z_86 += features[48] * 0.249826f;
    z_86 += features[49] * 0.146049f;
    z_86 += features[53] * -0.172676f;
    z_86 += features[54] * 0.255776f;
    z_86 += features[56] * -0.177470f;
    z_86 += features[57] * -0.121889f;
    z_86 += features[61] * 0.225169f;
    float out_86 = 0.177923f * z_86;
    {
        float arg_x = 0.078439f * z_86 + -0.081133f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.129821f * z_86 + 0.079832f) + 1e-6f;
        out_86 += 0.031873f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.073417f * z_86 + -0.026169f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.095490f * z_86 + 0.024028f) + 1e-6f;
        out_86 += 0.039385f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.073308f * z_86 + -0.060609f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.077038f * z_86 + 0.061191f) + 1e-6f;
        out_86 += 0.001852f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.057656f * z_86 + -0.067771f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.112837f * z_86 + 0.067078f) + 1e-6f;
        out_86 += 0.006421f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[86] = out_86;

    // Node 87
    float z_87 = 0.0f;
    z_87 += features[0] * 0.267451f;
    z_87 += features[1] * 0.210668f;
    z_87 += features[3] * 0.127565f;
    z_87 += features[4] * -0.114521f;
    z_87 += features[5] * 0.267145f;
    z_87 += features[6] * 0.189917f;
    z_87 += features[9] * 0.163520f;
    z_87 += features[10] * 0.122593f;
    z_87 += features[12] * 0.334510f;
    z_87 += features[15] * -0.145550f;
    z_87 += features[17] * -0.211740f;
    z_87 += features[19] * 0.280885f;
    z_87 += features[20] * -0.237615f;
    z_87 += features[21] * -0.118281f;
    z_87 += features[22] * -0.167768f;
    z_87 += features[23] * -0.142906f;
    z_87 += features[24] * 0.170710f;
    z_87 += features[25] * -0.231881f;
    z_87 += features[26] * 0.139917f;
    z_87 += features[27] * -0.223311f;
    z_87 += features[30] * -0.170936f;
    z_87 += features[31] * 0.117436f;
    z_87 += features[32] * 0.276838f;
    z_87 += features[33] * -0.141490f;
    z_87 += features[34] * 0.176551f;
    z_87 += features[35] * -0.207813f;
    z_87 += features[37] * 0.177012f;
    z_87 += features[38] * 0.110595f;
    z_87 += features[39] * 0.164879f;
    z_87 += features[41] * -0.200286f;
    z_87 += features[43] * 0.114297f;
    z_87 += features[44] * 0.218790f;
    z_87 += features[46] * -0.272774f;
    z_87 += features[47] * -0.248684f;
    z_87 += features[49] * 0.216089f;
    z_87 += features[50] * -0.213813f;
    z_87 += features[52] * 0.266313f;
    z_87 += features[54] * -0.205156f;
    z_87 += features[58] * -0.295772f;
    z_87 += features[60] * -0.117244f;
    float out_87 = 0.248435f * z_87;
    {
        float arg_x = -0.328990f * z_87 + 0.056788f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.246347f * z_87 + -0.021222f) + 1e-6f;
        out_87 += -0.068033f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.068424f * z_87 + -0.065587f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.006443f * z_87 + 0.059379f) + 1e-6f;
        out_87 += -0.012467f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.255099f * z_87 + 0.063594f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.228952f * z_87 + -0.041175f) + 1e-6f;
        out_87 += -0.096101f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.073943f * z_87 + -0.106861f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.060513f * z_87 + 0.107153f) + 1e-6f;
        out_87 += 0.011764f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[87] = out_87;

    // Node 88
    float z_88 = 0.0f;
    z_88 += features[1] * 0.207781f;
    z_88 += features[2] * 0.138992f;
    z_88 += features[3] * 0.186923f;
    z_88 += features[6] * -0.125230f;
    z_88 += features[7] * -0.125739f;
    z_88 += features[9] * 0.157866f;
    z_88 += features[11] * 0.255024f;
    z_88 += features[12] * 0.156739f;
    z_88 += features[13] * 0.118825f;
    z_88 += features[18] * -0.137527f;
    z_88 += features[20] * 0.216135f;
    z_88 += features[21] * -0.180087f;
    z_88 += features[23] * 0.213612f;
    z_88 += features[25] * 0.252409f;
    z_88 += features[26] * 0.196054f;
    z_88 += features[27] * -0.165033f;
    z_88 += features[29] * -0.118404f;
    z_88 += features[31] * 0.117131f;
    z_88 += features[32] * 0.124054f;
    z_88 += features[38] * 0.135618f;
    z_88 += features[42] * -0.132347f;
    z_88 += features[43] * 0.153886f;
    z_88 += features[44] * 0.161200f;
    z_88 += features[48] * -0.144399f;
    z_88 += features[49] * 0.219121f;
    z_88 += features[50] * -0.205508f;
    z_88 += features[54] * -0.187923f;
    z_88 += features[56] * 0.229932f;
    z_88 += features[57] * -0.173971f;
    z_88 += features[58] * 0.112189f;
    z_88 += features[60] * -0.244097f;
    z_88 += features[62] * -0.263770f;
    float out_88 = 0.224385f * z_88;
    {
        float arg_x = 0.097756f * z_88 + -0.016712f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.068159f * z_88 + 0.018345f) + 1e-6f;
        out_88 += 0.029305f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.125916f * z_88 + 0.007479f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.073894f * z_88 + 0.001788f) + 1e-6f;
        out_88 += 0.048275f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.066767f * z_88 + -0.046626f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.076335f * z_88 + 0.042757f) + 1e-6f;
        out_88 += 0.026802f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.106232f * z_88 + -0.048872f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.118198f * z_88 + 0.057947f) + 1e-6f;
        out_88 += -0.021412f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[88] = out_88;

    // Node 89
    float z_89 = 0.0f;
    z_89 += features[1] * 0.263215f;
    z_89 += features[2] * 0.201336f;
    z_89 += features[6] * -0.158049f;
    z_89 += features[7] * -0.179211f;
    z_89 += features[9] * -0.179099f;
    z_89 += features[10] * -0.254448f;
    z_89 += features[11] * -0.186533f;
    z_89 += features[12] * -0.144056f;
    z_89 += features[14] * 0.225755f;
    z_89 += features[15] * 0.115126f;
    z_89 += features[16] * -0.205010f;
    z_89 += features[18] * -0.217689f;
    z_89 += features[19] * 0.247743f;
    z_89 += features[26] * 0.178193f;
    z_89 += features[30] * -0.151472f;
    z_89 += features[31] * 0.151811f;
    z_89 += features[33] * -0.155464f;
    z_89 += features[35] * 0.119387f;
    z_89 += features[36] * 0.154482f;
    z_89 += features[37] * 0.148834f;
    z_89 += features[38] * -0.172627f;
    z_89 += features[39] * 0.139726f;
    z_89 += features[40] * 0.236510f;
    z_89 += features[43] * 0.114078f;
    z_89 += features[46] * -0.239564f;
    z_89 += features[47] * -0.230292f;
    z_89 += features[49] * 0.203841f;
    z_89 += features[52] * -0.230543f;
    z_89 += features[53] * 0.111354f;
    z_89 += features[54] * -0.146230f;
    z_89 += features[57] * -0.112724f;
    z_89 += features[58] * -0.129415f;
    z_89 += features[61] * -0.160797f;
    float out_89 = 0.193625f * z_89;
    {
        float arg_x = 0.108261f * z_89 + -0.047260f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.051067f * z_89 + 0.051088f) + 1e-6f;
        out_89 += 0.015818f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.090912f * z_89 + 0.003355f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.121563f * z_89 + -0.007752f) + 1e-6f;
        out_89 += 0.061047f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.041729f * z_89 + -0.095938f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.024707f * z_89 + 0.103198f) + 1e-6f;
        out_89 += -0.000999f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.094668f * z_89 + -0.008929f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.072033f * z_89 + 0.008690f) + 1e-6f;
        out_89 += 0.026130f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[89] = out_89;

    // Node 90
    float z_90 = 0.0f;
    z_90 += features[0] * 0.198958f;
    z_90 += features[2] * 0.119955f;
    z_90 += features[3] * 0.117229f;
    z_90 += features[4] * -0.174267f;
    z_90 += features[5] * 0.160267f;
    z_90 += features[6] * -0.135512f;
    z_90 += features[9] * -0.118252f;
    z_90 += features[10] * -0.188265f;
    z_90 += features[13] * -0.171860f;
    z_90 += features[16] * 0.135878f;
    z_90 += features[18] * -0.203032f;
    z_90 += features[19] * 0.112371f;
    z_90 += features[20] * 0.237810f;
    z_90 += features[22] * 0.235193f;
    z_90 += features[23] * -0.143122f;
    z_90 += features[27] * 0.181688f;
    z_90 += features[30] * 0.197948f;
    z_90 += features[31] * 0.129025f;
    z_90 += features[32] * -0.211859f;
    z_90 += features[33] * -0.156239f;
    z_90 += features[38] * -0.221262f;
    z_90 += features[39] * 0.188574f;
    z_90 += features[43] * -0.132595f;
    z_90 += features[45] * -0.208469f;
    z_90 += features[47] * -0.126743f;
    z_90 += features[50] * -0.202317f;
    z_90 += features[52] * -0.140851f;
    z_90 += features[53] * 0.254777f;
    z_90 += features[55] * -0.239768f;
    z_90 += features[56] * 0.235899f;
    z_90 += features[57] * 0.196050f;
    z_90 += features[59] * -0.154568f;
    z_90 += features[61] * -0.201972f;
    z_90 += features[62] * 0.215215f;
    float out_90 = 0.224453f * z_90;
    {
        float arg_x = -0.080540f * z_90 + -0.063341f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.090255f * z_90 + 0.067838f) + 1e-6f;
        out_90 += -0.004756f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.078171f * z_90 + -0.018450f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.090787f * z_90 + 0.017556f) + 1e-6f;
        out_90 += -0.028064f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.113861f * z_90 + 0.000923f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.143338f * z_90 + 0.012185f) + 1e-6f;
        out_90 += -0.038978f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.033284f * z_90 + -0.173751f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.003053f * z_90 + 0.174552f) + 1e-6f;
        out_90 += 0.021794f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[90] = out_90;

    // Node 91
    float z_91 = 0.0f;
    z_91 += features[0] * -0.201000f;
    z_91 += features[1] * -0.114689f;
    z_91 += features[4] * 0.222322f;
    z_91 += features[5] * -0.235257f;
    z_91 += features[7] * -0.142941f;
    z_91 += features[8] * 0.110242f;
    z_91 += features[12] * -0.225681f;
    z_91 += features[14] * -0.267909f;
    z_91 += features[16] * 0.127881f;
    z_91 += features[18] * 0.144389f;
    z_91 += features[19] * -0.260048f;
    z_91 += features[21] * 0.126427f;
    z_91 += features[22] * 0.205222f;
    z_91 += features[23] * 0.161388f;
    z_91 += features[24] * -0.166223f;
    z_91 += features[27] * -0.181293f;
    z_91 += features[31] * -0.240139f;
    z_91 += features[32] * -0.308467f;
    z_91 += features[34] * -0.185855f;
    z_91 += features[35] * -0.183098f;
    z_91 += features[37] * -0.186288f;
    z_91 += features[40] * -0.196025f;
    z_91 += features[41] * 0.191838f;
    z_91 += features[42] * 0.130391f;
    z_91 += features[43] * -0.176911f;
    z_91 += features[45] * 0.189654f;
    z_91 += features[47] * 0.292092f;
    z_91 += features[57] * 0.148756f;
    z_91 += features[58] * 0.144545f;
    z_91 += features[61] * 0.226787f;
    z_91 += features[62] * 0.195491f;
    z_91 += features[63] * -0.216851f;
    float out_91 = 0.235531f * z_91;
    {
        float arg_x = -0.174603f * z_91 + 0.057874f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.180903f * z_91 + -0.035226f) + 1e-6f;
        out_91 += -0.083351f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.126374f * z_91 + 0.025171f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.166101f * z_91 + -0.026310f) + 1e-6f;
        out_91 += -0.059075f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.133093f * z_91 + 0.027668f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.139623f * z_91 + -0.021200f) + 1e-6f;
        out_91 += -0.055560f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.062912f * z_91 + -0.111635f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.058776f * z_91 + 0.112403f) + 1e-6f;
        out_91 += 0.006045f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[91] = out_91;

    // Node 92
    float z_92 = 0.0f;
    z_92 += features[0] * 0.238887f;
    z_92 += features[3] * -0.189032f;
    z_92 += features[6] * 0.120607f;
    z_92 += features[7] * 0.290615f;
    z_92 += features[8] * -0.124540f;
    z_92 += features[9] * 0.212338f;
    z_92 += features[11] * 0.121378f;
    z_92 += features[12] * 0.282822f;
    z_92 += features[14] * 0.146176f;
    z_92 += features[19] * 0.219305f;
    z_92 += features[21] * -0.126951f;
    z_92 += features[22] * -0.142079f;
    z_92 += features[23] * -0.253656f;
    z_92 += features[24] * 0.355510f;
    z_92 += features[25] * -0.247130f;
    z_92 += features[27] * 0.213058f;
    z_92 += features[32] * 0.308236f;
    z_92 += features[33] * -0.240312f;
    z_92 += features[34] * 0.136043f;
    z_92 += features[37] * 0.146927f;
    z_92 += features[38] * 0.181386f;
    z_92 += features[41] * -0.210779f;
    z_92 += features[42] * -0.141796f;
    z_92 += features[43] * 0.142823f;
    z_92 += features[44] * 0.188441f;
    z_92 += features[45] * -0.144343f;
    z_92 += features[47] * -0.227753f;
    z_92 += features[49] * -0.333741f;
    z_92 += features[50] * -0.110059f;
    z_92 += features[52] * 0.223624f;
    z_92 += features[53] * -0.112142f;
    z_92 += features[54] * -0.149948f;
    z_92 += features[60] * 0.181060f;
    z_92 += features[61] * -0.232179f;
    z_92 += features[63] * 0.188675f;
    float out_92 = 0.301919f * z_92;
    {
        float arg_x = -0.410657f * z_92 + 0.009203f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.309273f * z_92 + 0.058726f) + 1e-6f;
        out_92 += -0.052152f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.281770f * z_92 + 0.053358f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.256904f * z_92 + -0.012594f) + 1e-6f;
        out_92 += 0.081188f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.306623f * z_92 + -0.022364f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.266710f * z_92 + 0.059157f) + 1e-6f;
        out_92 += -0.036668f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.363066f * z_92 + -0.012771f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.269470f * z_92 + 0.066063f) + 1e-6f;
        out_92 += -0.038859f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[92] = out_92;

    // Node 93
    float z_93 = 0.0f;
    z_93 += features[1] * -0.194007f;
    z_93 += features[2] * -0.180011f;
    z_93 += features[8] * -0.163490f;
    z_93 += features[10] * 0.190357f;
    z_93 += features[16] * 0.178173f;
    z_93 += features[17] * -0.158110f;
    z_93 += features[19] * -0.210113f;
    z_93 += features[21] * 0.190108f;
    z_93 += features[23] * 0.194483f;
    z_93 += features[25] * 0.246521f;
    z_93 += features[28] * 0.127532f;
    z_93 += features[31] * -0.179858f;
    z_93 += features[32] * 0.111354f;
    z_93 += features[33] * 0.143158f;
    z_93 += features[36] * -0.126480f;
    z_93 += features[38] * 0.257874f;
    z_93 += features[39] * -0.220460f;
    z_93 += features[40] * -0.132290f;
    z_93 += features[44] * 0.222687f;
    z_93 += features[45] * 0.164063f;
    z_93 += features[46] * 0.132621f;
    z_93 += features[47] * 0.215486f;
    z_93 += features[48] * 0.183960f;
    z_93 += features[50] * 0.148508f;
    z_93 += features[53] * -0.125141f;
    z_93 += features[54] * 0.217424f;
    z_93 += features[55] * 0.115025f;
    z_93 += features[57] * -0.118738f;
    z_93 += features[58] * 0.212514f;
    z_93 += features[59] * 0.194079f;
    z_93 += features[60] * 0.129479f;
    z_93 += features[63] * -0.125224f;
    float out_93 = 0.173871f * z_93;
    {
        float arg_x = 0.071111f * z_93 + 0.017486f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.062272f * z_93 + -0.015283f) + 1e-6f;
        out_93 += 0.062280f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.077259f * z_93 + 0.011028f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.059874f * z_93 + -0.004976f) + 1e-6f;
        out_93 += 0.047277f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.088723f * z_93 + -0.112371f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.113771f * z_93 + 0.113306f) + 1e-6f;
        out_93 += 0.011356f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.106406f * z_93 + -0.086103f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.104427f * z_93 + 0.088138f) + 1e-6f;
        out_93 += 0.017950f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[93] = out_93;

    // Node 94
    float z_94 = 0.0f;
    z_94 += features[1] * 0.132483f;
    z_94 += features[2] * 0.185522f;
    z_94 += features[4] * 0.205485f;
    z_94 += features[6] * -0.199553f;
    z_94 += features[7] * -0.143380f;
    z_94 += features[10] * -0.134458f;
    z_94 += features[11] * 0.211406f;
    z_94 += features[12] * -0.124787f;
    z_94 += features[14] * -0.258374f;
    z_94 += features[15] * 0.138415f;
    z_94 += features[19] * -0.228604f;
    z_94 += features[20] * 0.242047f;
    z_94 += features[21] * 0.285973f;
    z_94 += features[23] * -0.156948f;
    z_94 += features[25] * 0.143116f;
    z_94 += features[27] * 0.133643f;
    z_94 += features[28] * -0.197653f;
    z_94 += features[30] * 0.175532f;
    z_94 += features[32] * -0.274686f;
    z_94 += features[33] * -0.112686f;
    z_94 += features[37] * -0.246540f;
    z_94 += features[38] * -0.240037f;
    z_94 += features[40] * -0.149191f;
    z_94 += features[41] * 0.211704f;
    z_94 += features[44] * -0.252280f;
    z_94 += features[46] * 0.110077f;
    z_94 += features[47] * 0.201517f;
    z_94 += features[48] * -0.171874f;
    z_94 += features[50] * 0.132072f;
    z_94 += features[52] * -0.199630f;
    z_94 += features[53] * 0.203583f;
    z_94 += features[55] * 0.230475f;
    z_94 += features[56] * 0.173118f;
    z_94 += features[57] * 0.175921f;
    z_94 += features[59] * 0.147374f;
    z_94 += features[63] * -0.156378f;
    float out_94 = 0.227102f * z_94;
    {
        float arg_x = -0.136275f * z_94 + -0.057042f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.120976f * z_94 + 0.068945f) + 1e-6f;
        out_94 += 0.020775f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.119185f * z_94 + -0.035076f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.149287f * z_94 + 0.041068f) + 1e-6f;
        out_94 += -0.006761f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.122292f * z_94 + -0.049201f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.128041f * z_94 + 0.056378f) + 1e-6f;
        out_94 += 0.020737f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.172534f * z_94 + 0.046670f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.139803f * z_94 + -0.024996f) + 1e-6f;
        out_94 += 0.089252f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[94] = out_94;

    // Node 95
    float z_95 = 0.0f;
    z_95 += features[0] * -0.193909f;
    z_95 += features[1] * 0.185664f;
    z_95 += features[3] * 0.198301f;
    z_95 += features[4] * -0.164446f;
    z_95 += features[6] * -0.175922f;
    z_95 += features[11] * -0.151123f;
    z_95 += features[13] * 0.133664f;
    z_95 += features[14] * 0.119222f;
    z_95 += features[19] * 0.118907f;
    z_95 += features[20] * 0.162445f;
    z_95 += features[21] * -0.163275f;
    z_95 += features[22] * -0.162641f;
    z_95 += features[23] * 0.126104f;
    z_95 += features[25] * 0.171755f;
    z_95 += features[27] * -0.212453f;
    z_95 += features[29] * 0.140597f;
    z_95 += features[30] * -0.158580f;
    z_95 += features[31] * 0.136372f;
    z_95 += features[33] * 0.178379f;
    z_95 += features[35] * 0.227995f;
    z_95 += features[38] * 0.215052f;
    z_95 += features[40] * 0.221605f;
    z_95 += features[44] * 0.136713f;
    z_95 += features[46] * -0.158656f;
    z_95 += features[47] * -0.154760f;
    z_95 += features[48] * 0.114608f;
    z_95 += features[49] * 0.265598f;
    z_95 += features[50] * -0.196380f;
    z_95 += features[52] * -0.160955f;
    z_95 += features[56] * 0.240938f;
    z_95 += features[60] * -0.176981f;
    z_95 += features[61] * 0.120006f;
    float out_95 = 0.225492f * z_95;
    {
        float arg_x = 0.091060f * z_95 + -0.058030f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.139357f * z_95 + 0.054527f) + 1e-6f;
        out_95 += 0.001549f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.102329f * z_95 + -0.019861f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.092785f * z_95 + 0.022550f) + 1e-6f;
        out_95 += 0.017371f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.110664f * z_95 + -0.022177f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.111525f * z_95 + 0.029180f) + 1e-6f;
        out_95 += -0.039647f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.104308f * z_95 + -0.087428f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.114356f * z_95 + 0.088783f) + 1e-6f;
        out_95 += -0.003348f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[95] = out_95;

    // Node 96
    float z_96 = 0.0f;
    z_96 += features[2] * 0.195508f;
    z_96 += features[4] * -0.125270f;
    z_96 += features[5] * 0.193687f;
    z_96 += features[10] * -0.230305f;
    z_96 += features[16] * -0.170503f;
    z_96 += features[18] * -0.143751f;
    z_96 += features[19] * -0.113172f;
    z_96 += features[20] * 0.280959f;
    z_96 += features[22] * 0.217744f;
    z_96 += features[23] * -0.176304f;
    z_96 += features[24] * -0.172501f;
    z_96 += features[31] * 0.192402f;
    z_96 += features[32] * -0.215044f;
    z_96 += features[35] * 0.210033f;
    z_96 += features[37] * -0.175563f;
    z_96 += features[38] * -0.153215f;
    z_96 += features[39] * 0.181202f;
    z_96 += features[40] * 0.182482f;
    z_96 += features[41] * 0.154061f;
    z_96 += features[44] * -0.202028f;
    z_96 += features[50] * -0.216014f;
    z_96 += features[52] * -0.256230f;
    z_96 += features[54] * -0.248796f;
    z_96 += features[55] * -0.126872f;
    z_96 += features[56] * 0.141559f;
    z_96 += features[62] * 0.144122f;
    z_96 += features[63] * -0.191690f;
    float out_96 = 0.177391f * z_96;
    {
        float arg_x = -0.077757f * z_96 + -0.090510f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.074674f * z_96 + 0.087586f) + 1e-6f;
        out_96 += -0.008154f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.073371f * z_96 + -0.035559f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.067880f * z_96 + 0.032647f) + 1e-6f;
        out_96 += -0.040158f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.068157f * z_96 + -0.148818f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.085037f * z_96 + 0.146370f) + 1e-6f;
        out_96 += -0.002013f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.006913f * z_96 + -0.132063f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.030037f * z_96 + 0.130446f) + 1e-6f;
        out_96 += 0.019270f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[96] = out_96;

    // Node 97
    float z_97 = 0.0f;
    z_97 += features[0] * 0.112106f;
    z_97 += features[6] * 0.116112f;
    z_97 += features[8] * 0.228203f;
    z_97 += features[9] * 0.160983f;
    z_97 += features[13] * -0.181337f;
    z_97 += features[16] * -0.164177f;
    z_97 += features[17] * 0.129052f;
    z_97 += features[18] * 0.115353f;
    z_97 += features[19] * 0.181173f;
    z_97 += features[20] * -0.242377f;
    z_97 += features[21] * -0.229879f;
    z_97 += features[23] * -0.249855f;
    z_97 += features[25] * -0.156288f;
    z_97 += features[29] * -0.226599f;
    z_97 += features[30] * 0.218057f;
    z_97 += features[32] * -0.160734f;
    z_97 += features[37] * -0.119640f;
    z_97 += features[38] * -0.191906f;
    z_97 += features[39] * -0.115191f;
    z_97 += features[43] * -0.116589f;
    z_97 += features[45] * 0.121756f;
    z_97 += features[47] * -0.205301f;
    z_97 += features[50] * 0.206942f;
    z_97 += features[52] * 0.216881f;
    z_97 += features[53] * 0.162928f;
    z_97 += features[54] * 0.220565f;
    z_97 += features[55] * -0.226088f;
    z_97 += features[59] * -0.116704f;
    z_97 += features[62] * -0.185398f;
    float out_97 = 0.230010f * z_97;
    {
        float arg_x = 0.078716f * z_97 + -0.082258f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.068850f * z_97 + 0.090167f) + 1e-6f;
        out_97 += -0.014804f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.074553f * z_97 + -0.015571f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.062868f * z_97 + 0.012164f) + 1e-6f;
        out_97 += -0.018909f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.154511f * z_97 + 0.031025f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.088780f * z_97 + -0.025286f) + 1e-6f;
        out_97 += -0.038609f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.049920f * z_97 + -0.140680f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.092459f * z_97 + 0.136390f) + 1e-6f;
        out_97 += 0.010253f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[97] = out_97;

    // Node 98
    float z_98 = 0.0f;
    z_98 += features[0] * -0.165149f;
    z_98 += features[1] * 0.228670f;
    z_98 += features[2] * -0.144591f;
    z_98 += features[5] * -0.169138f;
    z_98 += features[8] * -0.155853f;
    z_98 += features[9] * 0.203762f;
    z_98 += features[10] * 0.254970f;
    z_98 += features[11] * -0.161058f;
    z_98 += features[13] * 0.162681f;
    z_98 += features[14] * 0.186193f;
    z_98 += features[15] * -0.228214f;
    z_98 += features[16] * -0.118790f;
    z_98 += features[19] * 0.157426f;
    z_98 += features[20] * 0.160274f;
    z_98 += features[22] * -0.213089f;
    z_98 += features[23] * 0.234482f;
    z_98 += features[25] * 0.205401f;
    z_98 += features[27] * -0.192414f;
    z_98 += features[28] * 0.233355f;
    z_98 += features[31] * -0.138216f;
    z_98 += features[32] * 0.197710f;
    z_98 += features[33] * 0.213636f;
    z_98 += features[37] * 0.175074f;
    z_98 += features[40] * 0.141233f;
    z_98 += features[41] * -0.217078f;
    z_98 += features[42] * -0.147892f;
    z_98 += features[44] * 0.213562f;
    z_98 += features[46] * -0.210888f;
    z_98 += features[47] * -0.257040f;
    z_98 += features[49] * 0.263853f;
    z_98 += features[53] * -0.169034f;
    z_98 += features[54] * -0.146067f;
    z_98 += features[56] * -0.121196f;
    z_98 += features[58] * 0.131140f;
    z_98 += features[61] * 0.191074f;
    float out_98 = 0.202889f * z_98;
    {
        float arg_x = 0.067167f * z_98 + -0.036901f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.083052f * z_98 + 0.037964f) + 1e-6f;
        out_98 += 0.002270f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.050116f * z_98 + -0.051434f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.101433f * z_98 + 0.038820f) + 1e-6f;
        out_98 += -0.002468f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.063994f * z_98 + -0.143448f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.059633f * z_98 + 0.145759f) + 1e-6f;
        out_98 += -0.025829f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.061002f * z_98 + -0.061429f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.073148f * z_98 + 0.060465f) + 1e-6f;
        out_98 += -0.011893f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[98] = out_98;

    // Node 99
    float z_99 = 0.0f;
    z_99 += features[1] * -0.230651f;
    z_99 += features[3] * -0.121658f;
    z_99 += features[9] * -0.189748f;
    z_99 += features[10] * -0.123548f;
    z_99 += features[11] * -0.257057f;
    z_99 += features[13] * -0.228519f;
    z_99 += features[17] * 0.144642f;
    z_99 += features[18] * 0.153970f;
    z_99 += features[19] * 0.119526f;
    z_99 += features[22] * 0.125705f;
    z_99 += features[23] * 0.130066f;
    z_99 += features[25] * -0.123195f;
    z_99 += features[26] * -0.207625f;
    z_99 += features[27] * -0.173460f;
    z_99 += features[33] * -0.135471f;
    z_99 += features[34] * -0.247581f;
    z_99 += features[36] * 0.133408f;
    z_99 += features[37] * 0.154103f;
    z_99 += features[38] * -0.162738f;
    z_99 += features[41] * 0.181789f;
    z_99 += features[42] * 0.255332f;
    z_99 += features[43] * -0.183702f;
    z_99 += features[45] * 0.162168f;
    z_99 += features[49] * -0.205153f;
    z_99 += features[50] * 0.179045f;
    z_99 += features[51] * 0.212269f;
    z_99 += features[52] * -0.136398f;
    z_99 += features[53] * -0.189875f;
    z_99 += features[54] * 0.244283f;
    z_99 += features[56] * -0.283509f;
    z_99 += features[58] * -0.242107f;
    z_99 += features[60] * 0.119383f;
    z_99 += features[62] * 0.213439f;
    float out_99 = 0.229489f * z_99;
    {
        float arg_x = 0.120053f * z_99 + -0.067123f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.130399f * z_99 + 0.073170f) + 1e-6f;
        out_99 += 0.004885f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.012213f * z_99 + -0.066926f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.020923f * z_99 + 0.063794f) + 1e-6f;
        out_99 += -0.029928f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.145822f * z_99 + -0.006575f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.172629f * z_99 + 0.019979f) + 1e-6f;
        out_99 += 0.035423f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.133230f * z_99 + -0.035163f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.107172f * z_99 + 0.049799f) + 1e-6f;
        out_99 += 0.007447f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[99] = out_99;

    // Node 100
    float z_100 = 0.0f;
    z_100 += features[10] * -0.215494f;
    z_100 += features[13] * 0.147373f;
    z_100 += features[15] * -0.191552f;
    z_100 += features[18] * -0.203653f;
    z_100 += features[19] * 0.142675f;
    z_100 += features[24] * 0.123712f;
    z_100 += features[26] * 0.194136f;
    z_100 += features[28] * -0.153549f;
    z_100 += features[29] * 0.296693f;
    z_100 += features[30] * -0.242759f;
    z_100 += features[31] * 0.110577f;
    z_100 += features[32] * 0.216469f;
    z_100 += features[35] * 0.192184f;
    z_100 += features[36] * 0.169925f;
    z_100 += features[45] * -0.243179f;
    z_100 += features[46] * -0.132123f;
    z_100 += features[48] * 0.162435f;
    z_100 += features[49] * 0.260216f;
    z_100 += features[50] * -0.190384f;
    z_100 += features[52] * -0.208798f;
    z_100 += features[53] * -0.170142f;
    z_100 += features[54] * -0.131099f;
    z_100 += features[59] * 0.150369f;
    z_100 += features[61] * 0.179288f;
    z_100 += features[62] * 0.222836f;
    z_100 += features[63] * -0.206456f;
    float out_100 = 0.217067f * z_100;
    {
        float arg_x = 0.089054f * z_100 + -0.013144f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.112947f * z_100 + 0.011221f) + 1e-6f;
        out_100 += 0.039311f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.000525f * z_100 + -0.159062f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.036033f * z_100 + 0.159332f) + 1e-6f;
        out_100 += -0.007513f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.124899f * z_100 + 0.041439f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.127719f * z_100 + -0.032293f) + 1e-6f;
        out_100 += 0.094083f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.089396f * z_100 + -0.011186f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.081707f * z_100 + 0.011146f) + 1e-6f;
        out_100 += 0.034468f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[100] = out_100;

    // Node 101
    float z_101 = 0.0f;
    z_101 += features[0] * -0.133499f;
    z_101 += features[1] * 0.179495f;
    z_101 += features[6] * -0.163412f;
    z_101 += features[7] * -0.204729f;
    z_101 += features[8] * -0.220269f;
    z_101 += features[9] * 0.269024f;
    z_101 += features[12] * 0.120309f;
    z_101 += features[13] * 0.113463f;
    z_101 += features[15] * -0.122706f;
    z_101 += features[16] * -0.144909f;
    z_101 += features[17] * -0.161809f;
    z_101 += features[19] * 0.126256f;
    z_101 += features[20] * 0.187776f;
    z_101 += features[21] * -0.184379f;
    z_101 += features[26] * 0.211290f;
    z_101 += features[30] * -0.196156f;
    z_101 += features[33] * 0.193903f;
    z_101 += features[34] * 0.234397f;
    z_101 += features[35] * 0.123596f;
    z_101 += features[39] * 0.174623f;
    z_101 += features[40] * 0.169695f;
    z_101 += features[41] * -0.140601f;
    z_101 += features[42] * -0.136710f;
    z_101 += features[43] * 0.122198f;
    z_101 += features[47] * -0.124051f;
    z_101 += features[49] * 0.190274f;
    z_101 += features[50] * -0.149507f;
    z_101 += features[56] * 0.120089f;
    z_101 += features[59] * 0.161993f;
    z_101 += features[61] * -0.235999f;
    z_101 += features[62] * -0.214871f;
    z_101 += features[63] * 0.157369f;
    float out_101 = 0.182271f * z_101;
    {
        float arg_x = -0.128750f * z_101 + -0.008992f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.099862f * z_101 + 0.025833f) + 1e-6f;
        out_101 += -0.056688f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.077602f * z_101 + -0.076789f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.117600f * z_101 + 0.073305f) + 1e-6f;
        out_101 += -0.036923f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.076103f * z_101 + 0.016656f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.099135f * z_101 + -0.021059f) + 1e-6f;
        out_101 += -0.060225f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.055427f * z_101 + 0.009778f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.072822f * z_101 + -0.019629f) + 1e-6f;
        out_101 += -0.051514f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[101] = out_101;

    // Node 102
    float z_102 = 0.0f;
    z_102 += features[2] * -0.233473f;
    z_102 += features[3] * 0.258714f;
    z_102 += features[4] * -0.122739f;
    z_102 += features[6] * 0.176958f;
    z_102 += features[13] * 0.203007f;
    z_102 += features[17] * -0.156266f;
    z_102 += features[22] * -0.160293f;
    z_102 += features[24] * 0.129818f;
    z_102 += features[27] * -0.141764f;
    z_102 += features[29] * 0.277724f;
    z_102 += features[30] * -0.161518f;
    z_102 += features[32] * 0.141523f;
    z_102 += features[33] * 0.272670f;
    z_102 += features[34] * -0.130582f;
    z_102 += features[35] * -0.165026f;
    z_102 += features[37] * -0.130984f;
    z_102 += features[38] * 0.122106f;
    z_102 += features[39] * -0.145080f;
    z_102 += features[41] * 0.198588f;
    z_102 += features[42] * 0.265228f;
    z_102 += features[43] * -0.195812f;
    z_102 += features[44] * 0.148140f;
    z_102 += features[45] * 0.139410f;
    z_102 += features[46] * 0.188380f;
    z_102 += features[48] * 0.152680f;
    z_102 += features[50] * -0.115457f;
    z_102 += features[55] * 0.145381f;
    z_102 += features[58] * -0.173774f;
    z_102 += features[61] * 0.131984f;
    float out_102 = 0.170784f * z_102;
    {
        float arg_x = 0.114734f * z_102 + -0.105115f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.080830f * z_102 + 0.105675f) + 1e-6f;
        out_102 += -0.010183f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.039030f * z_102 + -0.068528f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.062548f * z_102 + 0.066716f) + 1e-6f;
        out_102 += -0.013275f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.072722f * z_102 + -0.041656f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.098792f * z_102 + 0.039858f) + 1e-6f;
        out_102 += 0.019742f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.051161f * z_102 + -0.039787f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.079936f * z_102 + 0.036175f) + 1e-6f;
        out_102 += 0.023237f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[102] = out_102;

    // Node 103
    float z_103 = 0.0f;
    z_103 += features[2] * -0.154451f;
    z_103 += features[4] * -0.168865f;
    z_103 += features[5] * 0.123469f;
    z_103 += features[6] * 0.185674f;
    z_103 += features[7] * -0.120416f;
    z_103 += features[9] * 0.158196f;
    z_103 += features[11] * 0.153161f;
    z_103 += features[12] * 0.133387f;
    z_103 += features[13] * -0.161141f;
    z_103 += features[14] * 0.195361f;
    z_103 += features[15] * -0.134911f;
    z_103 += features[16] * -0.232951f;
    z_103 += features[19] * 0.163505f;
    z_103 += features[22] * -0.207581f;
    z_103 += features[23] * -0.125221f;
    z_103 += features[26] * 0.256676f;
    z_103 += features[29] * -0.251868f;
    z_103 += features[32] * 0.135795f;
    z_103 += features[33] * 0.124840f;
    z_103 += features[34] * 0.250252f;
    z_103 += features[38] * -0.120969f;
    z_103 += features[39] * 0.139391f;
    z_103 += features[40] * 0.123566f;
    z_103 += features[41] * -0.241380f;
    z_103 += features[42] * -0.125468f;
    z_103 += features[44] * 0.166551f;
    z_103 += features[48] * -0.205544f;
    z_103 += features[49] * 0.245815f;
    z_103 += features[50] * -0.159808f;
    z_103 += features[52] * 0.164121f;
    z_103 += features[53] * 0.198131f;
    z_103 += features[54] * -0.160160f;
    z_103 += features[55] * -0.167272f;
    z_103 += features[56] * -0.159309f;
    z_103 += features[59] * -0.127507f;
    z_103 += features[60] * -0.216709f;
    z_103 += features[61] * -0.222738f;
    z_103 += features[63] * 0.171236f;
    float out_103 = 0.203168f * z_103;
    {
        float arg_x = 0.142187f * z_103 + -0.030602f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.102310f * z_103 + 0.056290f) + 1e-6f;
        out_103 += -0.035061f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.136814f * z_103 + 0.091326f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.107095f * z_103 + -0.077242f) + 1e-6f;
        out_103 += -0.086416f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.101919f * z_103 + -0.063104f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.160285f * z_103 + 0.065967f) + 1e-6f;
        out_103 += -0.028280f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.076903f * z_103 + 0.067419f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.126580f * z_103 + -0.071247f) + 1e-6f;
        out_103 += -0.078879f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[103] = out_103;

    // Node 104
    float z_104 = 0.0f;
    z_104 += features[0] * -0.179457f;
    z_104 += features[1] * 0.130996f;
    z_104 += features[3] * 0.110717f;
    z_104 += features[4] * 0.156711f;
    z_104 += features[7] * -0.113460f;
    z_104 += features[10] * 0.198059f;
    z_104 += features[12] * 0.170374f;
    z_104 += features[15] * -0.126272f;
    z_104 += features[20] * 0.191960f;
    z_104 += features[21] * -0.139737f;
    z_104 += features[25] * 0.122232f;
    z_104 += features[26] * 0.245873f;
    z_104 += features[27] * -0.190484f;
    z_104 += features[32] * 0.165439f;
    z_104 += features[33] * 0.201791f;
    z_104 += features[38] * 0.152382f;
    z_104 += features[41] * -0.148196f;
    z_104 += features[42] * 0.148900f;
    z_104 += features[48] * 0.133354f;
    z_104 += features[52] * -0.120228f;
    z_104 += features[54] * -0.115451f;
    z_104 += features[57] * -0.210626f;
    z_104 += features[58] * 0.139220f;
    z_104 += features[59] * 0.124425f;
    z_104 += features[60] * -0.154665f;
    z_104 += features[62] * -0.144428f;
    z_104 += features[63] * -0.200816f;
    float out_104 = 0.150517f * z_104;
    {
        float arg_x = 0.112616f * z_104 + -0.035326f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.063563f * z_104 + 0.039473f) + 1e-6f;
        out_104 += -0.038281f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.078720f * z_104 + -0.045637f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.044194f * z_104 + 0.046843f) + 1e-6f;
        out_104 += -0.016652f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.024681f * z_104 + 0.012198f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.031615f * z_104 + -0.018228f) + 1e-6f;
        out_104 += -0.057140f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.011327f * z_104 + -0.020410f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.030895f * z_104 + 0.011390f) + 1e-6f;
        out_104 += -0.045246f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[104] = out_104;

    // Node 105
    float z_105 = 0.0f;
    z_105 += features[0] * -0.211501f;
    z_105 += features[1] * -0.130898f;
    z_105 += features[2] * -0.195278f;
    z_105 += features[4] * -0.305964f;
    z_105 += features[7] * 0.201187f;
    z_105 += features[11] * -0.256729f;
    z_105 += features[13] * 0.224762f;
    z_105 += features[15] * -0.117851f;
    z_105 += features[16] * 0.199099f;
    z_105 += features[17] * -0.155253f;
    z_105 += features[19] * 0.130822f;
    z_105 += features[20] * -0.136404f;
    z_105 += features[21] * -0.111596f;
    z_105 += features[24] * 0.140502f;
    z_105 += features[29] * 0.281670f;
    z_105 += features[30] * -0.265005f;
    z_105 += features[32] * 0.204990f;
    z_105 += features[34] * -0.150165f;
    z_105 += features[35] * 0.211966f;
    z_105 += features[38] * 0.171763f;
    z_105 += features[44] * 0.258033f;
    z_105 += features[45] * -0.292513f;
    z_105 += features[48] * 0.256046f;
    z_105 += features[53] * -0.244990f;
    z_105 += features[56] * -0.173135f;
    z_105 += features[57] * -0.152359f;
    z_105 += features[58] * 0.165590f;
    z_105 += features[63] * 0.164153f;
    float out_105 = 0.272751f * z_105;
    {
        float arg_x = -0.064656f * z_105 + -0.168239f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.087976f * z_105 + 0.167676f) + 1e-6f;
        out_105 += -0.024717f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.149162f * z_105 + 0.000701f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.177046f * z_105 + 0.013885f) + 1e-6f;
        out_105 += 0.049217f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.076728f * z_105 + -0.106708f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.079537f * z_105 + 0.110725f) + 1e-6f;
        out_105 += 0.002608f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.155195f * z_105 + -0.018993f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.141776f * z_105 + 0.042227f) + 1e-6f;
        out_105 += 0.031461f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[105] = out_105;

    // Node 106
    float z_106 = 0.0f;
    z_106 += features[0] * 0.228528f;
    z_106 += features[1] * -0.182565f;
    z_106 += features[2] * -0.142741f;
    z_106 += features[3] * 0.202193f;
    z_106 += features[6] * 0.222736f;
    z_106 += features[10] * 0.123196f;
    z_106 += features[12] * 0.184366f;
    z_106 += features[13] * -0.215552f;
    z_106 += features[19] * -0.117662f;
    z_106 += features[20] * -0.150310f;
    z_106 += features[21] * 0.126404f;
    z_106 += features[30] * 0.151174f;
    z_106 += features[34] * -0.140922f;
    z_106 += features[35] * -0.220775f;
    z_106 += features[39] * -0.161166f;
    z_106 += features[41] * 0.172516f;
    z_106 += features[44] * -0.174360f;
    z_106 += features[45] * 0.228364f;
    z_106 += features[48] * -0.182339f;
    z_106 += features[49] * -0.175615f;
    z_106 += features[51] * -0.186100f;
    z_106 += features[52] * 0.150652f;
    z_106 += features[53] * 0.181825f;
    z_106 += features[54] * 0.236061f;
    z_106 += features[56] * -0.145657f;
    z_106 += features[58] * -0.164200f;
    z_106 += features[59] * -0.165901f;
    z_106 += features[60] * 0.143350f;
    z_106 += features[62] * 0.164761f;
    float out_106 = 0.203756f * z_106;
    {
        float arg_x = -0.146603f * z_106 + 0.029025f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.107036f * z_106 + -0.016722f) + 1e-6f;
        out_106 += -0.093123f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.114169f * z_106 + 0.014096f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.095762f * z_106 + -0.010021f) + 1e-6f;
        out_106 += -0.048529f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.069933f * z_106 + -0.016347f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.093057f * z_106 + 0.006217f) + 1e-6f;
        out_106 += -0.022087f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.056274f * z_106 + -0.071145f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.065193f * z_106 + 0.078237f) + 1e-6f;
        out_106 += 0.017846f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[106] = out_106;

    // Node 107
    float z_107 = 0.0f;
    z_107 += features[0] * 0.159481f;
    z_107 += features[1] * -0.199770f;
    z_107 += features[4] * 0.142909f;
    z_107 += features[6] * 0.125854f;
    z_107 += features[7] * 0.205466f;
    z_107 += features[11] * 0.155182f;
    z_107 += features[12] * 0.193000f;
    z_107 += features[13] * 0.131780f;
    z_107 += features[14] * -0.150482f;
    z_107 += features[15] * 0.174265f;
    z_107 += features[16] * 0.145668f;
    z_107 += features[19] * -0.189481f;
    z_107 += features[21] * 0.166918f;
    z_107 += features[22] * 0.125582f;
    z_107 += features[23] * -0.195118f;
    z_107 += features[24] * -0.110765f;
    z_107 += features[25] * 0.135927f;
    z_107 += features[27] * 0.114403f;
    z_107 += features[29] * -0.256159f;
    z_107 += features[35] * -0.143522f;
    z_107 += features[36] * -0.234865f;
    z_107 += features[40] * -0.227937f;
    z_107 += features[41] * -0.224852f;
    z_107 += features[42] * -0.231282f;
    z_107 += features[43] * 0.151047f;
    z_107 += features[45] * -0.130421f;
    z_107 += features[46] * 0.185969f;
    z_107 += features[47] * 0.232239f;
    z_107 += features[48] * -0.145657f;
    z_107 += features[49] * -0.160957f;
    z_107 += features[54] * -0.274416f;
    z_107 += features[55] * 0.120927f;
    z_107 += features[56] * 0.246016f;
    z_107 += features[57] * 0.162566f;
    z_107 += features[58] * 0.272233f;
    z_107 += features[61] * 0.111612f;
    z_107 += features[62] * -0.150252f;
    float out_107 = 0.222648f * z_107;
    {
        float arg_x = -0.080379f * z_107 + -0.017149f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.120501f * z_107 + 0.012925f) + 1e-6f;
        out_107 += -0.031683f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.106870f * z_107 + -0.047724f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.097360f * z_107 + 0.053775f) + 1e-6f;
        out_107 += -0.021592f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.064162f * z_107 + -0.083790f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.085979f * z_107 + 0.078846f) + 1e-6f;
        out_107 += -0.008752f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.070774f * z_107 + -0.025837f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.117267f * z_107 + 0.019444f) + 1e-6f;
        out_107 += -0.024993f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[107] = out_107;

    // Node 108
    float z_108 = 0.0f;
    z_108 += features[1] * -0.194652f;
    z_108 += features[4] * 0.187580f;
    z_108 += features[7] * 0.218722f;
    z_108 += features[8] * -0.119573f;
    z_108 += features[10] * 0.115444f;
    z_108 += features[12] * 0.179732f;
    z_108 += features[13] * 0.285452f;
    z_108 += features[14] * -0.246831f;
    z_108 += features[16] * 0.232328f;
    z_108 += features[18] * 0.207162f;
    z_108 += features[19] * -0.212351f;
    z_108 += features[20] * -0.222290f;
    z_108 += features[21] * 0.220059f;
    z_108 += features[23] * 0.141088f;
    z_108 += features[24] * -0.111286f;
    z_108 += features[25] * 0.244023f;
    z_108 += features[26] * -0.189560f;
    z_108 += features[27] * 0.135823f;
    z_108 += features[28] * 0.219898f;
    z_108 += features[29] * -0.191553f;
    z_108 += features[30] * -0.136145f;
    z_108 += features[31] * -0.164931f;
    z_108 += features[35] * -0.166793f;
    z_108 += features[36] * -0.275328f;
    z_108 += features[37] * -0.116201f;
    z_108 += features[38] * 0.196099f;
    z_108 += features[39] * -0.189505f;
    z_108 += features[40] * -0.213550f;
    z_108 += features[41] * -0.312021f;
    z_108 += features[42] * -0.283123f;
    z_108 += features[43] * 0.231781f;
    z_108 += features[44] * 0.136458f;
    z_108 += features[46] * 0.229101f;
    z_108 += features[47] * 0.253058f;
    z_108 += features[50] * -0.113001f;
    z_108 += features[51] * -0.213658f;
    z_108 += features[52] * 0.217982f;
    z_108 += features[54] * -0.248712f;
    z_108 += features[55] * 0.124431f;
    z_108 += features[57] * 0.130357f;
    z_108 += features[58] * 0.187169f;
    z_108 += features[60] * -0.136778f;
    z_108 += features[62] * -0.132807f;
    float out_108 = 0.247320f * z_108;
    {
        float arg_x = -0.278035f * z_108 + -0.013848f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.128757f * z_108 + 0.048525f) + 1e-6f;
        out_108 += -0.042068f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.146226f * z_108 + -0.044901f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.131933f * z_108 + 0.073444f) + 1e-6f;
        out_108 += 0.040133f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.259415f * z_108 + 0.092486f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.159312f * z_108 + -0.030506f) + 1e-6f;
        out_108 += -0.146302f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.126361f * z_108 + -0.103283f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.147310f * z_108 + 0.116425f) + 1e-6f;
        out_108 += 0.031123f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[108] = out_108;

    // Node 109
    float z_109 = 0.0f;
    z_109 += features[0] * 0.239342f;
    z_109 += features[2] * 0.194353f;
    z_109 += features[3] * -0.144018f;
    z_109 += features[5] * 0.286294f;
    z_109 += features[9] * 0.197574f;
    z_109 += features[11] * 0.221189f;
    z_109 += features[13] * -0.194690f;
    z_109 += features[15] * 0.173888f;
    z_109 += features[16] * -0.141898f;
    z_109 += features[17] * 0.174884f;
    z_109 += features[20] * -0.239306f;
    z_109 += features[23] * -0.131041f;
    z_109 += features[25] * -0.230141f;
    z_109 += features[26] * -0.163690f;
    z_109 += features[28] * -0.145603f;
    z_109 += features[29] * -0.138167f;
    z_109 += features[30] * 0.159981f;
    z_109 += features[31] * 0.149888f;
    z_109 += features[32] * -0.226598f;
    z_109 += features[33] * -0.183167f;
    z_109 += features[34] * 0.174004f;
    z_109 += features[36] * -0.194527f;
    z_109 += features[40] * -0.148640f;
    z_109 += features[41] * -0.292523f;
    z_109 += features[42] * -0.166782f;
    z_109 += features[43] * 0.167615f;
    z_109 += features[44] * 0.153586f;
    z_109 += features[45] * -0.222483f;
    z_109 += features[47] * 0.194027f;
    z_109 += features[51] * -0.134417f;
    z_109 += features[52] * 0.278254f;
    z_109 += features[53] * 0.290693f;
    z_109 += features[54] * -0.187875f;
    z_109 += features[55] * -0.283487f;
    z_109 += features[56] * 0.133100f;
    z_109 += features[59] * -0.219453f;
    z_109 += features[60] * 0.159125f;
    z_109 += features[62] * -0.188261f;
    z_109 += features[63] * 0.145410f;
    float out_109 = 0.251226f * z_109;
    {
        float arg_x = 0.128820f * z_109 + -0.072418f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.114557f * z_109 + 0.092299f) + 1e-6f;
        out_109 += 0.014953f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.139267f * z_109 + -0.012620f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.132927f * z_109 + 0.029384f) + 1e-6f;
        out_109 += 0.041652f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.231615f * z_109 + 0.030736f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.198852f * z_109 + 0.023717f) + 1e-6f;
        out_109 += 0.073108f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.158072f * z_109 + -0.000893f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.156085f * z_109 + 0.024243f) + 1e-6f;
        out_109 += 0.055503f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[109] = out_109;

    // Node 110
    float z_110 = 0.0f;
    z_110 += features[1] * -0.211787f;
    z_110 += features[2] * -0.244505f;
    z_110 += features[3] * -0.184448f;
    z_110 += features[4] * 0.144316f;
    z_110 += features[5] * -0.167623f;
    z_110 += features[6] * 0.161170f;
    z_110 += features[8] * 0.161608f;
    z_110 += features[9] * -0.215256f;
    z_110 += features[10] * -0.117733f;
    z_110 += features[11] * -0.131117f;
    z_110 += features[12] * -0.247652f;
    z_110 += features[13] * -0.188886f;
    z_110 += features[14] * -0.119408f;
    z_110 += features[15] * 0.135402f;
    z_110 += features[17] * 0.184725f;
    z_110 += features[18] * 0.133628f;
    z_110 += features[19] * -0.140655f;
    z_110 += features[20] * -0.124260f;
    z_110 += features[21] * 0.286290f;
    z_110 += features[22] * 0.222899f;
    z_110 += features[27] * -0.226506f;
    z_110 += features[28] * 0.138140f;
    z_110 += features[30] * 0.222904f;
    z_110 += features[31] * -0.166978f;
    z_110 += features[32] * -0.172412f;
    z_110 += features[34] * -0.183838f;
    z_110 += features[35] * -0.185119f;
    z_110 += features[38] * -0.143217f;
    z_110 += features[39] * -0.138641f;
    z_110 += features[40] * -0.141069f;
    z_110 += features[41] * 0.278579f;
    z_110 += features[43] * -0.133889f;
    z_110 += features[45] * 0.139847f;
    z_110 += features[49] * -0.207243f;
    z_110 += features[50] * 0.164380f;
    z_110 += features[51] * -0.163837f;
    z_110 += features[54] * 0.169368f;
    z_110 += features[57] * 0.167036f;
    z_110 += features[58] * -0.110010f;
    z_110 += features[59] * -0.125959f;
    z_110 += features[60] * 0.168525f;
    z_110 += features[61] * 0.234168f;
    z_110 += features[63] * -0.212706f;
    float out_110 = 0.238410f * z_110;
    {
        float arg_x = -0.067585f * z_110 + -0.112609f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.103716f * z_110 + 0.107570f) + 1e-6f;
        out_110 += 0.008087f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.101331f * z_110 + -0.046628f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.099314f * z_110 + 0.051821f) + 1e-6f;
        out_110 += -0.010477f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.169327f * z_110 + 0.029378f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.137460f * z_110 + 0.006491f) + 1e-6f;
        out_110 += -0.080405f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.020927f * z_110 + -0.111577f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.003867f * z_110 + 0.110134f) + 1e-6f;
        out_110 += 0.017328f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[110] = out_110;

    // Node 111
    float z_111 = 0.0f;
    z_111 += features[1] * -0.206534f;
    z_111 += features[6] * 0.163877f;
    z_111 += features[7] * 0.182195f;
    z_111 += features[10] * 0.152054f;
    z_111 += features[11] * -0.152816f;
    z_111 += features[12] * 0.237487f;
    z_111 += features[16] * 0.209417f;
    z_111 += features[18] * 0.199705f;
    z_111 += features[19] * 0.200352f;
    z_111 += features[20] * -0.257598f;
    z_111 += features[22] * -0.154024f;
    z_111 += features[25] * -0.158082f;
    z_111 += features[28] * 0.109766f;
    z_111 += features[30] * 0.194335f;
    z_111 += features[31] * -0.204355f;
    z_111 += features[32] * 0.196159f;
    z_111 += features[34] * -0.161577f;
    z_111 += features[36] * -0.183087f;
    z_111 += features[44] * 0.318420f;
    z_111 += features[49] * -0.139458f;
    z_111 += features[51] * -0.136031f;
    z_111 += features[52] * 0.195440f;
    z_111 += features[56] * -0.263134f;
    z_111 += features[58] * -0.184542f;
    z_111 += features[59] * -0.140810f;
    z_111 += features[60] * 0.124519f;
    float out_111 = 0.207511f * z_111;
    {
        float arg_x = -0.000274f * z_111 + -0.068470f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.032259f * z_111 + 0.063976f) + 1e-6f;
        out_111 += -0.037782f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.103136f * z_111 + -0.022639f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.109787f * z_111 + 0.033769f) + 1e-6f;
        out_111 += 0.033544f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.069544f * z_111 + -0.046627f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.061245f * z_111 + 0.048621f) + 1e-6f;
        out_111 += 0.006369f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.047398f * z_111 + -0.029135f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.116885f * z_111 + 0.023605f) + 1e-6f;
        out_111 += -0.049598f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[111] = out_111;

    // Node 112
    float z_112 = 0.0f;
    z_112 += features[4] * 0.228210f;
    z_112 += features[6] * -0.132895f;
    z_112 += features[7] * -0.236363f;
    z_112 += features[10] * -0.140282f;
    z_112 += features[12] * 0.128660f;
    z_112 += features[14] * -0.153280f;
    z_112 += features[19] * -0.217908f;
    z_112 += features[21] * 0.231122f;
    z_112 += features[22] * 0.134722f;
    z_112 += features[23] * 0.226068f;
    z_112 += features[24] * 0.117653f;
    z_112 += features[25] * 0.216438f;
    z_112 += features[27] * -0.266867f;
    z_112 += features[30] * -0.131671f;
    z_112 += features[34] * -0.230885f;
    z_112 += features[38] * -0.238472f;
    z_112 += features[39] * -0.110640f;
    z_112 += features[42] * 0.179550f;
    z_112 += features[47] * 0.227071f;
    z_112 += features[48] * 0.125838f;
    z_112 += features[49] * 0.257099f;
    z_112 += features[50] * -0.114693f;
    z_112 += features[52] * -0.180888f;
    z_112 += features[53] * -0.218346f;
    z_112 += features[55] * 0.154340f;
    z_112 += features[56] * 0.222209f;
    z_112 += features[59] * 0.229523f;
    z_112 += features[63] * -0.275588f;
    float out_112 = 0.214495f * z_112;
    {
        float arg_x = 0.188533f * z_112 + 0.055827f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.112875f * z_112 + -0.018291f) + 1e-6f;
        out_112 += 0.076928f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.200747f * z_112 + 0.062216f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.094736f * z_112 + -0.017673f) + 1e-6f;
        out_112 += 0.082153f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.144652f * z_112 + 0.029649f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.126790f * z_112 + -0.014016f) + 1e-6f;
        out_112 += 0.055033f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.160187f * z_112 + 0.045525f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.151007f * z_112 + -0.025373f) + 1e-6f;
        out_112 += 0.095957f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[112] = out_112;

    // Node 113
    float z_113 = 0.0f;
    z_113 += features[0] * -0.152305f;
    z_113 += features[3] * -0.142883f;
    z_113 += features[4] * 0.115668f;
    z_113 += features[5] * -0.248290f;
    z_113 += features[7] * 0.149265f;
    z_113 += features[8] * -0.224590f;
    z_113 += features[10] * 0.199562f;
    z_113 += features[11] * 0.138661f;
    z_113 += features[12] * 0.153785f;
    z_113 += features[13] * 0.215126f;
    z_113 += features[14] * -0.133227f;
    z_113 += features[16] * 0.180315f;
    z_113 += features[17] * -0.236559f;
    z_113 += features[18] * 0.115606f;
    z_113 += features[19] * -0.243303f;
    z_113 += features[21] * 0.209712f;
    z_113 += features[23] * 0.221399f;
    z_113 += features[25] * 0.202927f;
    z_113 += features[28] * 0.180519f;
    z_113 += features[29] * -0.114085f;
    z_113 += features[31] * -0.184249f;
    z_113 += features[35] * -0.153195f;
    z_113 += features[36] * -0.216212f;
    z_113 += features[37] * 0.113410f;
    z_113 += features[38] * 0.153922f;
    z_113 += features[41] * -0.202754f;
    z_113 += features[42] * -0.132019f;
    z_113 += features[43] * 0.186233f;
    z_113 += features[44] * 0.203357f;
    z_113 += features[45] * -0.241023f;
    z_113 += features[46] * 0.138678f;
    z_113 += features[47] * 0.154510f;
    z_113 += features[49] * -0.235790f;
    z_113 += features[50] * 0.202674f;
    z_113 += features[51] * 0.180020f;
    z_113 += features[52] * 0.146272f;
    z_113 += features[53] * -0.299098f;
    z_113 += features[54] * 0.159603f;
    z_113 += features[55] * 0.247135f;
    z_113 += features[56] * -0.120771f;
    z_113 += features[58] * 0.210331f;
    float out_113 = 0.225615f * z_113;
    {
        float arg_x = 0.103779f * z_113 + -0.220253f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.122485f * z_113 + 0.217653f) + 1e-6f;
        out_113 += 0.000746f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.117014f * z_113 + -0.070212f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.128402f * z_113 + 0.078417f) + 1e-6f;
        out_113 += 0.029673f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.170249f * z_113 + 0.024587f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.144051f * z_113 + 0.043022f) + 1e-6f;
        out_113 += 0.067157f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.139042f * z_113 + -0.004396f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.128789f * z_113 + 0.036807f) + 1e-6f;
        out_113 += 0.063427f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[113] = out_113;

    // Node 114
    float z_114 = 0.0f;
    z_114 += features[2] * -0.164879f;
    z_114 += features[3] * 0.234729f;
    z_114 += features[8] * 0.188269f;
    z_114 += features[10] * -0.263723f;
    z_114 += features[11] * -0.180738f;
    z_114 += features[12] * 0.219829f;
    z_114 += features[13] * 0.232155f;
    z_114 += features[14] * -0.172378f;
    z_114 += features[15] * -0.149962f;
    z_114 += features[17] * 0.232987f;
    z_114 += features[19] * -0.183049f;
    z_114 += features[20] * 0.154952f;
    z_114 += features[21] * 0.283890f;
    z_114 += features[24] * -0.267919f;
    z_114 += features[25] * 0.145109f;
    z_114 += features[27] * -0.173574f;
    z_114 += features[30] * 0.200234f;
    z_114 += features[31] * -0.191342f;
    z_114 += features[32] * -0.119553f;
    z_114 += features[33] * 0.186062f;
    z_114 += features[34] * -0.144942f;
    z_114 += features[37] * -0.117926f;
    z_114 += features[41] * 0.238904f;
    z_114 += features[42] * 0.279651f;
    z_114 += features[43] * -0.155577f;
    z_114 += features[45] * 0.172964f;
    z_114 += features[46] * 0.224841f;
    z_114 += features[47] * -0.136386f;
    z_114 += features[48] * -0.134117f;
    z_114 += features[49] * 0.291914f;
    z_114 += features[51] * -0.241824f;
    z_114 += features[55] * -0.112521f;
    z_114 += features[56] * 0.155365f;
    z_114 += features[59] * -0.194748f;
    z_114 += features[60] * -0.242847f;
    z_114 += features[62] * 0.179773f;
    z_114 += features[63] * -0.173855f;
    float out_114 = 0.241363f * z_114;
    {
        float arg_x = 0.046223f * z_114 + -0.016596f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.056118f * z_114 + 0.012267f) + 1e-6f;
        out_114 += 0.053984f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.202836f * z_114 + 0.003153f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.170738f * z_114 + 0.036433f) + 1e-6f;
        out_114 += -0.065911f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.071313f * z_114 + -0.007126f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.074348f * z_114 + 0.025708f) + 1e-6f;
        out_114 += 0.052520f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.240083f * z_114 + -0.000917f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.125965f * z_114 + 0.060160f) + 1e-6f;
        out_114 += -0.021508f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[114] = out_114;

    // Node 115
    float z_115 = 0.0f;
    z_115 += features[1] * 0.206799f;
    z_115 += features[3] * 0.163158f;
    z_115 += features[4] * -0.139024f;
    z_115 += features[5] * -0.172035f;
    z_115 += features[6] * -0.140158f;
    z_115 += features[7] * -0.263250f;
    z_115 += features[10] * -0.132272f;
    z_115 += features[11] * -0.227298f;
    z_115 += features[12] * -0.214278f;
    z_115 += features[17] * 0.200009f;
    z_115 += features[19] * 0.219854f;
    z_115 += features[20] * 0.112313f;
    z_115 += features[21] * 0.121548f;
    z_115 += features[26] * 0.284476f;
    z_115 += features[27] * -0.122073f;
    z_115 += features[32] * -0.319801f;
    z_115 += features[33] * 0.222309f;
    z_115 += features[34] * -0.133493f;
    z_115 += features[37] * -0.186308f;
    z_115 += features[39] * 0.174766f;
    z_115 += features[40] * 0.222972f;
    z_115 += features[42] * 0.277865f;
    z_115 += features[45] * 0.138298f;
    z_115 += features[49] * 0.126620f;
    z_115 += features[53] * 0.196668f;
    z_115 += features[54] * -0.271857f;
    z_115 += features[55] * -0.160278f;
    z_115 += features[57] * 0.180891f;
    z_115 += features[59] * -0.178746f;
    z_115 += features[60] * -0.223769f;
    z_115 += features[61] * -0.221630f;
    float out_115 = 0.220625f * z_115;
    {
        float arg_x = -0.055481f * z_115 + -0.075084f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.055052f * z_115 + 0.068400f) + 1e-6f;
        out_115 += -0.033990f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.165959f * z_115 + 0.069625f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.169286f * z_115 + -0.061008f) + 1e-6f;
        out_115 += -0.096123f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.118529f * z_115 + -0.006836f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.138200f * z_115 + 0.004028f) + 1e-6f;
        out_115 += -0.061831f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.200074f * z_115 + 0.075138f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.129268f * z_115 + -0.052107f) + 1e-6f;
        out_115 += -0.088664f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[115] = out_115;

    // Node 116
    float z_116 = 0.0f;
    z_116 += features[0] * -0.190299f;
    z_116 += features[3] * -0.167609f;
    z_116 += features[5] * -0.236636f;
    z_116 += features[6] * -0.183121f;
    z_116 += features[7] * 0.129460f;
    z_116 += features[8] * -0.204150f;
    z_116 += features[10] * 0.201518f;
    z_116 += features[11] * -0.148197f;
    z_116 += features[13] * 0.220833f;
    z_116 += features[16] * 0.153055f;
    z_116 += features[17] * -0.210289f;
    z_116 += features[21] * -0.121794f;
    z_116 += features[23] * 0.134651f;
    z_116 += features[24] * -0.215089f;
    z_116 += features[27] * 0.176046f;
    z_116 += features[30] * -0.229460f;
    z_116 += features[31] * 0.121329f;
    z_116 += features[32] * 0.210351f;
    z_116 += features[38] * 0.227230f;
    z_116 += features[39] * 0.201069f;
    z_116 += features[41] * -0.126420f;
    z_116 += features[44] * 0.153712f;
    z_116 += features[45] * -0.204008f;
    z_116 += features[49] * 0.120458f;
    z_116 += features[52] * -0.212989f;
    z_116 += features[53] * -0.242273f;
    z_116 += features[55] * 0.249829f;
    z_116 += features[56] * 0.126030f;
    z_116 += features[59] * 0.241114f;
    z_116 += features[63] * 0.134414f;
    float out_116 = 0.218551f * z_116;
    {
        float arg_x = 0.020913f * z_116 + -0.118607f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.031131f * z_116 + 0.116791f) + 1e-6f;
        out_116 += -0.019857f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.147262f * z_116 + 0.003371f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.145486f * z_116 + 0.015659f) + 1e-6f;
        out_116 += 0.041472f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.168872f * z_116 + 0.025099f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.122955f * z_116 + 0.013016f) + 1e-6f;
        out_116 += 0.058673f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.109630f * z_116 + -0.032272f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.098490f * z_116 + 0.036478f) + 1e-6f;
        out_116 += 0.016736f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[116] = out_116;

    // Node 117
    float z_117 = 0.0f;
    z_117 += features[0] * -0.140790f;
    z_117 += features[4] * 0.116509f;
    z_117 += features[6] * 0.131544f;
    z_117 += features[7] * -0.204078f;
    z_117 += features[9] * 0.153851f;
    z_117 += features[14] * 0.217608f;
    z_117 += features[16] * -0.265865f;
    z_117 += features[19] * 0.234266f;
    z_117 += features[21] * -0.261329f;
    z_117 += features[22] * -0.127787f;
    z_117 += features[24] * -0.142145f;
    z_117 += features[26] * 0.225649f;
    z_117 += features[28] * 0.160744f;
    z_117 += features[29] * -0.231952f;
    z_117 += features[30] * -0.187464f;
    z_117 += features[32] * 0.180789f;
    z_117 += features[33] * 0.162762f;
    z_117 += features[36] * -0.113915f;
    z_117 += features[38] * 0.175137f;
    z_117 += features[40] * 0.116588f;
    z_117 += features[41] * -0.271271f;
    z_117 += features[42] * -0.165793f;
    z_117 += features[44] * 0.243637f;
    z_117 += features[45] * 0.215737f;
    z_117 += features[46] * -0.256087f;
    z_117 += features[49] * 0.247827f;
    z_117 += features[50] * -0.145080f;
    z_117 += features[52] * 0.253507f;
    z_117 += features[53] * -0.114442f;
    z_117 += features[56] * -0.252900f;
    z_117 += features[57] * -0.151241f;
    z_117 += features[58] * 0.174796f;
    z_117 += features[59] * -0.114685f;
    z_117 += features[60] * -0.129903f;
    z_117 += features[61] * -0.146524f;
    z_117 += features[62] * -0.233303f;
    z_117 += features[63] * 0.216439f;
    float out_117 = 0.229269f * z_117;
    {
        float arg_x = 0.112500f * z_117 + -0.035936f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.099195f * z_117 + 0.048188f) + 1e-6f;
        out_117 += -0.012536f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.110294f * z_117 + 0.051440f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.115483f * z_117 + -0.044121f) + 1e-6f;
        out_117 += -0.080373f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.116415f * z_117 + -0.025846f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.115777f * z_117 + 0.039800f) + 1e-6f;
        out_117 += 0.020401f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.115074f * z_117 + -0.032776f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.124275f * z_117 + 0.046564f) + 1e-6f;
        out_117 += 0.002918f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[117] = out_117;

    // Node 118
    float z_118 = 0.0f;
    z_118 += features[0] * 0.239381f;
    z_118 += features[8] * -0.134830f;
    z_118 += features[9] * -0.124836f;
    z_118 += features[10] * -0.200876f;
    z_118 += features[11] * 0.184645f;
    z_118 += features[12] * 0.221625f;
    z_118 += features[13] * 0.154052f;
    z_118 += features[14] * -0.270846f;
    z_118 += features[16] * 0.268365f;
    z_118 += features[17] * -0.135783f;
    z_118 += features[20] * 0.126019f;
    z_118 += features[21] * 0.258205f;
    z_118 += features[24] * -0.202024f;
    z_118 += features[25] * 0.208833f;
    z_118 += features[28] * -0.169428f;
    z_118 += features[31] * 0.151037f;
    z_118 += features[32] * -0.210657f;
    z_118 += features[33] * -0.163147f;
    z_118 += features[36] * -0.203757f;
    z_118 += features[37] * -0.251039f;
    z_118 += features[40] * -0.267721f;
    z_118 += features[44] * 0.143456f;
    z_118 += features[45] * -0.172580f;
    z_118 += features[46] * 0.131794f;
    z_118 += features[47] * 0.210898f;
    z_118 += features[50] * -0.171297f;
    z_118 += features[52] * -0.160313f;
    z_118 += features[54] * -0.194404f;
    z_118 += features[56] * 0.259443f;
    z_118 += features[57] * 0.134352f;
    z_118 += features[61] * 0.149280f;
    z_118 += features[63] * -0.268899f;
    float out_118 = 0.237745f * z_118;
    {
        float arg_x = -0.120350f * z_118 + -0.086191f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.139664f * z_118 + 0.088591f) + 1e-6f;
        out_118 += 0.013065f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.131976f * z_118 + -0.078138f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.111032f * z_118 + 0.083834f) + 1e-6f;
        out_118 += 0.013357f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.057513f * z_118 + 0.000419f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.027787f * z_118 + -0.000167f) + 1e-6f;
        out_118 += 0.074755f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.008132f * z_118 + -0.047491f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.037507f * z_118 + 0.039774f) + 1e-6f;
        out_118 += 0.062547f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[118] = out_118;

    // Node 119
    float z_119 = 0.0f;
    z_119 += features[0] * -0.154152f;
    z_119 += features[3] * -0.172505f;
    z_119 += features[6] * -0.191732f;
    z_119 += features[8] * -0.144509f;
    z_119 += features[12] * 0.307388f;
    z_119 += features[13] * 0.153486f;
    z_119 += features[14] * -0.123203f;
    z_119 += features[16] * 0.145597f;
    z_119 += features[17] * -0.121682f;
    z_119 += features[21] * 0.230227f;
    z_119 += features[23] * 0.257290f;
    z_119 += features[24] * 0.248732f;
    z_119 += features[27] * -0.269062f;
    z_119 += features[29] * 0.230782f;
    z_119 += features[30] * -0.249607f;
    z_119 += features[32] * 0.245093f;
    z_119 += features[36] * -0.129637f;
    z_119 += features[38] * 0.134301f;
    z_119 += features[40] * -0.156155f;
    z_119 += features[41] * -0.136763f;
    z_119 += features[43] * 0.215959f;
    z_119 += features[44] * 0.246987f;
    z_119 += features[45] * -0.284948f;
    z_119 += features[48] * 0.241953f;
    z_119 += features[50] * -0.194516f;
    z_119 += features[53] * -0.183655f;
    z_119 += features[54] * -0.186486f;
    z_119 += features[55] * 0.175175f;
    z_119 += features[59] * 0.177365f;
    z_119 += features[61] * 0.219365f;
    z_119 += features[62] * 0.144403f;
    z_119 += features[63] * -0.263522f;
    float out_119 = 0.219354f * z_119;
    {
        float arg_x = 0.101104f * z_119 + -0.079336f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.112408f * z_119 + 0.072172f) + 1e-6f;
        out_119 += -0.009773f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.034929f * z_119 + -0.047392f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.050841f * z_119 + 0.047912f) + 1e-6f;
        out_119 += -0.028187f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.134039f * z_119 + -0.011507f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.118484f * z_119 + 0.023051f) + 1e-6f;
        out_119 += 0.033812f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.168703f * z_119 + 0.012362f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.136695f * z_119 + 0.017030f) + 1e-6f;
        out_119 += 0.024602f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[119] = out_119;

    // Node 120
    float z_120 = 0.0f;
    z_120 += features[2] * -0.203271f;
    z_120 += features[6] * 0.185515f;
    z_120 += features[7] * 0.190057f;
    z_120 += features[10] * 0.135933f;
    z_120 += features[13] * -0.129577f;
    z_120 += features[14] * -0.203779f;
    z_120 += features[15] * -0.197541f;
    z_120 += features[16] * 0.162117f;
    z_120 += features[17] * -0.148615f;
    z_120 += features[19] * 0.128372f;
    z_120 += features[20] * -0.222764f;
    z_120 += features[24] * 0.196292f;
    z_120 += features[27] * 0.139733f;
    z_120 += features[28] * 0.228166f;
    z_120 += features[29] * 0.157486f;
    z_120 += features[34] * -0.141072f;
    z_120 += features[36] * -0.181443f;
    z_120 += features[38] * 0.134419f;
    z_120 += features[40] * -0.232757f;
    z_120 += features[48] * 0.256440f;
    z_120 += features[49] * -0.225189f;
    z_120 += features[50] * 0.134284f;
    z_120 += features[52] * 0.159313f;
    z_120 += features[53] * -0.111785f;
    z_120 += features[56] * -0.243177f;
    z_120 += features[59] * -0.230207f;
    z_120 += features[60] * 0.222729f;
    z_120 += features[61] * 0.171025f;
    z_120 += features[63] * 0.232604f;
    float out_120 = 0.215018f * z_120;
    {
        float arg_x = 0.103461f * z_120 + -0.145546f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.049915f * z_120 + 0.150680f) + 1e-6f;
        out_120 += -0.007176f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.103780f * z_120 + -0.118525f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.137538f * z_120 + 0.117982f) + 1e-6f;
        out_120 += 0.008807f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.083670f * z_120 + -0.009386f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.072529f * z_120 + 0.008421f) + 1e-6f;
        out_120 += 0.053491f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.122449f * z_120 + -0.060559f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.085088f * z_120 + 0.063685f) + 1e-6f;
        out_120 += 0.013237f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[120] = out_120;

    // Node 121
    float z_121 = 0.0f;
    z_121 += features[2] * -0.162971f;
    z_121 += features[4] * 0.183390f;
    z_121 += features[5] * 0.151994f;
    z_121 += features[6] * -0.222236f;
    z_121 += features[7] * -0.313242f;
    z_121 += features[8] * 0.254022f;
    z_121 += features[9] * -0.130100f;
    z_121 += features[12] * -0.149942f;
    z_121 += features[14] * -0.190642f;
    z_121 += features[16] * -0.158450f;
    z_121 += features[17] * 0.229114f;
    z_121 += features[21] * 0.248938f;
    z_121 += features[23] * 0.157447f;
    z_121 += features[24] * -0.145432f;
    z_121 += features[25] * 0.164210f;
    z_121 += features[27] * -0.216422f;
    z_121 += features[30] * 0.128885f;
    z_121 += features[31] * -0.178491f;
    z_121 += features[32] * -0.250937f;
    z_121 += features[36] * 0.151697f;
    z_121 += features[37] * -0.169385f;
    z_121 += features[38] * -0.295186f;
    z_121 += features[39] * -0.126268f;
    z_121 += features[42] * 0.265228f;
    z_121 += features[45] * 0.200077f;
    z_121 += features[46] * 0.237086f;
    z_121 += features[47] * 0.269670f;
    z_121 += features[49] * 0.231398f;
    z_121 += features[50] * 0.258802f;
    z_121 += features[51] * 0.195209f;
    z_121 += features[53] * 0.224763f;
    z_121 += features[56] * 0.182217f;
    z_121 += features[58] * -0.121523f;
    z_121 += features[60] * -0.180049f;
    z_121 += features[61] * 0.170146f;
    z_121 += features[63] * -0.293697f;
    float out_121 = 0.232460f * z_121;
    {
        float arg_x = 0.260664f * z_121 + 0.073070f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.191975f * z_121 + -0.025460f) + 1e-6f;
        out_121 += 0.099428f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.253045f * z_121 + 0.116867f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.176013f * z_121 + -0.070422f) + 1e-6f;
        out_121 += 0.114766f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.166766f * z_121 + -0.004208f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.125763f * z_121 + 0.020146f) + 1e-6f;
        out_121 += 0.060119f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.190807f * z_121 + 0.018228f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.146966f * z_121 + 0.006309f) + 1e-6f;
        out_121 += 0.070480f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[121] = out_121;

    // Node 122
    float z_122 = 0.0f;
    z_122 += features[1] * -0.156003f;
    z_122 += features[2] * -0.126899f;
    z_122 += features[3] * -0.115201f;
    z_122 += features[6] * 0.131378f;
    z_122 += features[7] * 0.191970f;
    z_122 += features[8] * 0.190358f;
    z_122 += features[11] * -0.202873f;
    z_122 += features[12] * -0.275265f;
    z_122 += features[14] * -0.246492f;
    z_122 += features[16] * 0.270027f;
    z_122 += features[18] * 0.117665f;
    z_122 += features[19] * -0.157306f;
    z_122 += features[21] * 0.169447f;
    z_122 += features[22] * 0.226061f;
    z_122 += features[23] * -0.155178f;
    z_122 += features[24] * -0.205430f;
    z_122 += features[26] * -0.253165f;
    z_122 += features[27] * 0.177035f;
    z_122 += features[29] * -0.181800f;
    z_122 += features[30] * 0.176836f;
    z_122 += features[31] * -0.156745f;
    z_122 += features[35] * -0.218514f;
    z_122 += features[37] * -0.136235f;
    z_122 += features[38] * 0.154121f;
    z_122 += features[40] * -0.165190f;
    z_122 += features[41] * 0.273683f;
    z_122 += features[43] * -0.229465f;
    z_122 += features[44] * -0.162534f;
    z_122 += features[45] * 0.132828f;
    z_122 += features[46] * 0.149896f;
    z_122 += features[47] * 0.200632f;
    z_122 += features[48] * -0.218690f;
    z_122 += features[49] * -0.174838f;
    z_122 += features[50] * 0.223469f;
    z_122 += features[54] * 0.283446f;
    z_122 += features[58] * 0.144631f;
    z_122 += features[59] * -0.131935f;
    z_122 += features[60] * 0.159092f;
    z_122 += features[61] * 0.130340f;
    z_122 += features[63] * 0.162659f;
    float out_122 = 0.253596f * z_122;
    {
        float arg_x = -0.140729f * z_122 + -0.011166f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.174931f * z_122 + 0.016404f) + 1e-6f;
        out_122 += -0.033929f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.038015f * z_122 + -0.103049f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.023242f * z_122 + 0.101600f) + 1e-6f;
        out_122 += 0.012253f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.184550f * z_122 + 0.028560f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.193795f * z_122 + -0.011705f) + 1e-6f;
        out_122 += -0.061368f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.120098f * z_122 + -0.010021f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.185864f * z_122 + -0.010001f) + 1e-6f;
        out_122 += -0.040925f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[122] = out_122;

    // Node 123
    float z_123 = 0.0f;
    z_123 += features[0] * -0.186062f;
    z_123 += features[3] * -0.114519f;
    z_123 += features[4] * -0.250922f;
    z_123 += features[5] * -0.169143f;
    z_123 += features[12] * -0.222963f;
    z_123 += features[15] * -0.134331f;
    z_123 += features[16] * 0.144940f;
    z_123 += features[17] * -0.201076f;
    z_123 += features[19] * 0.246411f;
    z_123 += features[21] * -0.157467f;
    z_123 += features[22] * -0.176757f;
    z_123 += features[25] * 0.148438f;
    z_123 += features[29] * 0.187001f;
    z_123 += features[30] * -0.131024f;
    z_123 += features[31] * 0.184038f;
    z_123 += features[32] * 0.117039f;
    z_123 += features[33] * 0.162348f;
    z_123 += features[35] * 0.230516f;
    z_123 += features[36] * 0.157498f;
    z_123 += features[38] * 0.126281f;
    z_123 += features[39] * 0.135671f;
    z_123 += features[42] * -0.171654f;
    z_123 += features[46] * -0.134079f;
    z_123 += features[47] * -0.228696f;
    z_123 += features[48] * 0.236864f;
    z_123 += features[50] * -0.225710f;
    z_123 += features[51] * 0.223619f;
    z_123 += features[52] * -0.152492f;
    z_123 += features[54] * -0.204789f;
    z_123 += features[55] * 0.150215f;
    z_123 += features[57] * -0.139708f;
    z_123 += features[59] * 0.168865f;
    z_123 += features[60] * -0.137619f;
    z_123 += features[63] * 0.155884f;
    float out_123 = 0.229456f * z_123;
    {
        float arg_x = 0.161739f * z_123 + 0.021199f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.156838f * z_123 + -0.006004f) + 1e-6f;
        out_123 += 0.077411f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.106889f * z_123 + -0.004003f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.150344f * z_123 + 0.001148f) + 1e-6f;
        out_123 += 0.042835f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.104545f * z_123 + -0.072892f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.127475f * z_123 + 0.071484f) + 1e-6f;
        out_123 += 0.007680f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.132644f * z_123 + 0.009952f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.171754f * z_123 + -0.005773f) + 1e-6f;
        out_123 += 0.068276f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[123] = out_123;

    // Node 124
    float z_124 = 0.0f;
    z_124 += features[0] * 0.162691f;
    z_124 += features[1] * -0.152939f;
    z_124 += features[2] * -0.187148f;
    z_124 += features[6] * 0.217040f;
    z_124 += features[7] * -0.132366f;
    z_124 += features[8] * 0.185417f;
    z_124 += features[10] * -0.160681f;
    z_124 += features[11] * 0.215401f;
    z_124 += features[13] * -0.152599f;
    z_124 += features[14] * -0.220675f;
    z_124 += features[15] * 0.242087f;
    z_124 += features[18] * 0.223541f;
    z_124 += features[19] * -0.214042f;
    z_124 += features[21] * 0.267893f;
    z_124 += features[22] * 0.204878f;
    z_124 += features[24] * -0.266223f;
    z_124 += features[25] * -0.118989f;
    z_124 += features[27] * 0.155570f;
    z_124 += features[28] * 0.117387f;
    z_124 += features[29] * -0.169143f;
    z_124 += features[31] * -0.229439f;
    z_124 += features[32] * -0.117288f;
    z_124 += features[35] * -0.237046f;
    z_124 += features[36] * -0.211592f;
    z_124 += features[38] * -0.269813f;
    z_124 += features[41] * 0.149362f;
    z_124 += features[42] * 0.152178f;
    z_124 += features[43] * -0.153105f;
    z_124 += features[44] * -0.127265f;
    z_124 += features[47] * 0.139559f;
    z_124 += features[48] * -0.187536f;
    z_124 += features[49] * -0.117339f;
    z_124 += features[50] * 0.266673f;
    z_124 += features[51] * -0.195988f;
    z_124 += features[52] * 0.136028f;
    z_124 += features[54] * 0.169205f;
    z_124 += features[58] * -0.149565f;
    z_124 += features[59] * -0.109576f;
    float out_124 = 0.230098f * z_124;
    {
        float arg_x = -0.198502f * z_124 + 0.057809f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.138587f * z_124 + -0.019516f) + 1e-6f;
        out_124 += -0.100473f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.172459f * z_124 + 0.041597f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.120033f * z_124 + -0.014263f) + 1e-6f;
        out_124 += -0.069839f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.168655f * z_124 + 0.015129f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.111690f * z_124 + 0.009419f) + 1e-6f;
        out_124 += -0.044094f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.178142f * z_124 + 0.042286f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.141668f * z_124 + -0.015074f) + 1e-6f;
        out_124 += -0.065981f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[124] = out_124;

    // Node 125
    float z_125 = 0.0f;
    z_125 += features[2] * -0.135803f;
    z_125 += features[3] * 0.165537f;
    z_125 += features[4] * 0.267717f;
    z_125 += features[5] * -0.173091f;
    z_125 += features[10] * -0.234037f;
    z_125 += features[12] * -0.171463f;
    z_125 += features[16] * 0.176266f;
    z_125 += features[17] * 0.110906f;
    z_125 += features[18] * 0.202193f;
    z_125 += features[21] * 0.293834f;
    z_125 += features[22] * 0.194128f;
    z_125 += features[27] * 0.204511f;
    z_125 += features[28] * 0.129381f;
    z_125 += features[29] * -0.142058f;
    z_125 += features[30] * 0.132370f;
    z_125 += features[31] * -0.160279f;
    z_125 += features[32] * -0.279085f;
    z_125 += features[33] * 0.129033f;
    z_125 += features[34] * -0.247056f;
    z_125 += features[35] * -0.165347f;
    z_125 += features[42] * 0.237333f;
    z_125 += features[43] * -0.207950f;
    z_125 += features[44] * -0.186839f;
    z_125 += features[45] * 0.200708f;
    z_125 += features[48] * -0.210044f;
    z_125 += features[49] * -0.171609f;
    z_125 += features[50] * 0.173062f;
    z_125 += features[53] * 0.158189f;
    z_125 += features[55] * 0.154552f;
    z_125 += features[56] * 0.141900f;
    z_125 += features[57] * 0.200585f;
    z_125 += features[61] * -0.138603f;
    z_125 += features[63] * 0.141158f;
    float out_125 = 0.192111f * z_125;
    {
        float arg_x = -0.116555f * z_125 + 0.014842f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.087183f * z_125 + -0.008386f) + 1e-6f;
        out_125 += -0.062930f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.060814f * z_125 + -0.062687f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.028154f * z_125 + 0.070425f) + 1e-6f;
        out_125 += 0.028128f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.162695f * z_125 + 0.031518f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.085836f * z_125 + -0.006151f) + 1e-6f;
        out_125 += -0.055434f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.007712f * z_125 + -0.090669f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.039994f * z_125 + 0.093377f) + 1e-6f;
        out_125 += 0.007589f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[125] = out_125;

    // Node 126
    float z_126 = 0.0f;
    z_126 += features[1] * -0.142156f;
    z_126 += features[2] * -0.114536f;
    z_126 += features[3] * -0.110086f;
    z_126 += features[4] * 0.213373f;
    z_126 += features[5] * -0.132156f;
    z_126 += features[6] * 0.200205f;
    z_126 += features[8] * 0.206868f;
    z_126 += features[13] * -0.144127f;
    z_126 += features[14] * -0.170801f;
    z_126 += features[15] * 0.149631f;
    z_126 += features[18] * 0.133027f;
    z_126 += features[19] * -0.230689f;
    z_126 += features[25] * -0.171278f;
    z_126 += features[27] * 0.151629f;
    z_126 += features[31] * -0.244086f;
    z_126 += features[32] * -0.259102f;
    z_126 += features[34] * -0.181517f;
    z_126 += features[36] * -0.182464f;
    z_126 += features[37] * -0.171995f;
    z_126 += features[40] * -0.113410f;
    z_126 += features[41] * 0.136357f;
    z_126 += features[42] * 0.223329f;
    z_126 += features[44] * -0.135678f;
    z_126 += features[45] * 0.188661f;
    z_126 += features[49] * -0.163248f;
    z_126 += features[50] * 0.172014f;
    z_126 += features[53] * 0.146894f;
    z_126 += features[54] * 0.227537f;
    z_126 += features[56] * -0.133287f;
    z_126 += features[57] * 0.166548f;
    z_126 += features[58] * -0.209714f;
    z_126 += features[59] * -0.159586f;
    z_126 += features[60] * 0.191728f;
    float out_126 = 0.190552f * z_126;
    {
        float arg_x = -0.079081f * z_126 + -0.039170f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.097624f * z_126 + 0.029753f) + 1e-6f;
        out_126 += -0.008633f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.064823f * z_126 + -0.053660f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.043118f * z_126 + 0.065597f) + 1e-6f;
        out_126 += 0.032367f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = -0.099110f * z_126 + -0.030648f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(0.120165f * z_126 + 0.024138f) + 1e-6f;
        out_126 += -0.015120f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.008440f * z_126 + -0.138231f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.043068f * z_126 + 0.134651f) + 1e-6f;
        out_126 += 0.015985f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[126] = out_126;

    // Node 127
    float z_127 = 0.0f;
    z_127 += features[4] * -0.114939f;
    z_127 += features[6] * 0.141185f;
    z_127 += features[10] * -0.162108f;
    z_127 += features[11] * -0.209230f;
    z_127 += features[13] * -0.284806f;
    z_127 += features[14] * -0.158211f;
    z_127 += features[17] * -0.152402f;
    z_127 += features[19] * 0.205506f;
    z_127 += features[20] * -0.175072f;
    z_127 += features[25] * -0.187293f;
    z_127 += features[29] * 0.159493f;
    z_127 += features[34] * -0.111205f;
    z_127 += features[36] * 0.127423f;
    z_127 += features[37] * 0.163104f;
    z_127 += features[41] * 0.256424f;
    z_127 += features[42] * 0.251062f;
    z_127 += features[43] * -0.128091f;
    z_127 += features[49] * -0.125577f;
    z_127 += features[50] * 0.165292f;
    z_127 += features[53] * -0.258633f;
    z_127 += features[54] * 0.195337f;
    z_127 += features[55] * 0.154740f;
    z_127 += features[56] * -0.168162f;
    z_127 += features[59] * -0.192073f;
    z_127 += features[60] * 0.185384f;
    z_127 += features[61] * 0.111930f;
    z_127 += features[62] * 0.161783f;
    float out_127 = 0.200746f * z_127;
    {
        float arg_x = 0.102130f * z_127 + -0.042382f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.101987f * z_127 + 0.044604f) + 1e-6f;
        out_127 += 0.001583f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.093068f * z_127 + -0.052952f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.112440f * z_127 + 0.053479f) + 1e-6f;
        out_127 += -0.014090f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.088840f * z_127 + -0.049252f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.110433f * z_127 + 0.049319f) + 1e-6f;
        out_127 += -0.009006f * (expf(arg_x) - logf(arg_y));
    }
    {
        float arg_x = 0.122976f * z_127 + -0.045439f;
        if (arg_x < -10.0f) arg_x = -10.0f;
        if (arg_x > 10.0f) arg_x = 10.0f;
        float arg_y = softplus_stable(-0.105876f * z_127 + 0.050408f) + 1e-6f;
        out_127 += -0.006141f * (expf(arg_x) - logf(arg_y));
    }
    output_logits[127] = out_127;

}

#endif // EML_KAN_LLM_DAG_H
