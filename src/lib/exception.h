/**
 * \file
 * \brief   A generic exception implementation, derived from std::exception and
 *          capable of transporting user-defined error messages
 *
 * \author  Georg Kuschk 12/2012
 */

#ifndef MUMFORDSHAH_LIB_EXCEPTION_H_
#define MUMFORDSHAH_LIB_EXCEPTION_H_


//STL&Co
#include <exception>
#include <stdarg.h>
#include <stdio.h>
#include <string>


// Macros for compact exception handling
// Example usage:
//   CATCH_EXIT( pImg = loadImage( "image.pgm" ); )

// Catch an exception, print to console and exit program with error value
#define CATCH_EXIT( x )                 \
    try { x }                       \
    catch ( Exception ex )          \
    { LOG( "%s\n", ex.what() );  \
    exit(1); }

// Catch an exception and throw it (upwards)
#define CATCH_THROW( x )                \
    try { x }                       \
    catch ( Exception ex )          \
    { throw ex; }



/**
 * \brief   A generic exception implementation, derived from std::exception and
 *          capable of transporting user-defined error messages
 *
 * \author  Georg Kuschk 12/2012
 */
class Exception: public std::exception
{
  public:

    /**
     * \brief  Default constructor
     */
    Exception( void ): std::exception()
    {
      m_szDescription = "Default exception";
    }


    /**
     * \brief  Constructor
     *
     * \param  szDescription The error message to throw
     */
    Exception( std::string szDescription ): std::exception()
    {
      m_szDescription = szDescription;
    }



    /**
     * \brief  Constructor
     *
     * \param  szFMT The format string followed by the variable list of arguments
     */
    Exception( const char *szFMT, ... ): std::exception()
    {
      char  str[512];

      va_list  argptr;
      va_start( argptr, szFMT );
      vsnprintf( str, 512, szFMT, argptr );
      va_end( argptr );

      m_szDescription = std::string( str );
    }



    /**
     * \brief  Destructor
     */
    virtual ~Exception( void ) throw()
        {
        }


    /**
     * \brief  Overridden function to get the error message
     *
     * \return The error message as a char-string
     */
    virtual char const *what() const throw()
        {
      return m_szDescription.c_str();
        }


  protected:

    //! This string contains the error message
    std::string   m_szDescription;

};


#endif //MUMFORDSHAH_LIB_EXCEPTION_H_
