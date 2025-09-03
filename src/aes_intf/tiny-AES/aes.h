// #define AES128 1
//#define AES192 1
//#define AES256 1

#define AES_BLOCKLEN 16 // Block length in bytes - AES is 128b block only
//    // Key length in bytes
#define AES_keyExpSize 176

// #define AES_KEYLEN 16
// #ifdef AES128 && AES128 == 1
// #endif
// // AES128 && AES128 == 1

// #ifdef AES192 && AES192 == 1
//   #define AES_KEYLEN 24
// #endif
// // AES192 && AES192 == 1

// #ifdef AES256 && AES256 == 1
//   #define AES_KEYLEN 32
// #endif
// // AES256 && AES256 == 1
  
// typedef uint8_t aes_key_t[AES_KEYLEN];

struct AES_ctx
{
  uint8_t RoundKey[AES_keyExpSize];
};

void AES_init_ctx(struct AES_ctx* ctx, const uint8_t* key);

void target(const struct AES_ctx* ctx, uint8_t* buf, int target_round, int target_step);

